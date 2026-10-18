//! Batched forward pieces: weights are read once for up to MAX_BATCH tokens.
//! Every kernel keeps the per-token arithmetic of the single-token path, so a
//! batched pass is bit-identical to running the tokens one by one.
use super::failure;
use adaforge_application::DiagnosticError;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::sync::Arc;

pub(crate) const MAX_BATCH: usize = 8;

pub(crate) struct BatchKernels {
    stream: Arc<CudaStream>,
    permute: CudaFunction,
    matvec: Vec<CudaFunction>,
    kparallel: Vec<CudaFunction>,
    tiled_gate: Vec<CudaFunction>,
    rms: CudaFunction,
    rms_add_rms: CudaFunction,
    rms_residual_scale: CudaFunction,
    gelu: CudaFunction,
}

fn grid(blocks: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn check(n: u32, len: usize, need: usize) -> Result<(), DiagnosticError> {
    if n == 0 || n as usize > MAX_BATCH || len < need {
        return Err(DiagnosticError("Lot invalide".into()));
    }
    Ok(())
}

impl BatchKernels {
    pub(crate) fn new(stream: Arc<CudaStream>) -> Result<Self, DiagnosticError> {
        let source = format!(
            "{}\n{}\n{}\n{}\n{}",
            include_str!("../../../kernels/ada/q4_matvec.cu"),
            include_str!("../../../kernels/ada/q4_tiled_probe.cu"),
            include_str!("../../../kernels/ada/q4_batch.cu"),
            include_str!("../../../kernels/ada/rms_norm.cu"),
            include_str!("../../../kernels/ada/ffn.cu"),
        );
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| failure("Compilation lots", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| failure("Module lots", e))?;
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| failure("Kernel lots", e))
        };
        let family = |prefix: &str| {
            (1..=MAX_BATCH)
                .map(|b| load(&format!("{prefix}{b}")))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            permute: load("q4_permute_x")?,
            matvec: family("q4_batch_matvec_b")?,
            kparallel: family("q4_batch_kparallel_b")?,
            tiled_gate: family("q4_batch_tiled_gate_b")?,
            rms: load("rms_norm")?,
            rms_add_rms: load("rms_add_rms")?,
            rms_residual_scale: load("rms_residual_scale")?,
            gelu: load("gelu_gate")?,
            stream,
        })
    }
    /// x [n][columns] -> permuted copy for the batched Q4 kernels.
    pub(crate) fn permute(
        &self,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        total: u32,
    ) -> Result<(), DiagnosticError> {
        if x.len() < total as usize || out.len() < total as usize || !total.is_multiple_of(32) {
            return Err(DiagnosticError("Permutation invalide".into()));
        }
        let mut l = self.stream.launch_builder(&self.permute);
        l.arg(x).arg(out).arg(&total);
        // SAFETY: total elements checked in both buffers, bounded threads.
        unsafe { l.launch(grid(total.div_ceil(256), 256)) }
            .map_err(|e| failure("Permutation lot", e))?;
        Ok(())
    }
    /// y[t][rows] = W x[t] for t < n; x permuted. kparallel selects the 4-warp tree.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matvec(
        &self,
        weights: &CudaSlice<u8>,
        xp: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        columns: u32,
        rows: u32,
        n: u32,
        kparallel: bool,
    ) -> Result<(), DiagnosticError> {
        check(n, y.len(), (n * rows) as usize)?;
        if xp.len() < (n * columns) as usize
            || weights.len() != (rows * columns / 32 * 18) as usize
        {
            return Err(DiagnosticError("Matvec lot invalide".into()));
        }
        let f = if kparallel {
            &self.kparallel[n as usize - 1]
        } else {
            &self.matvec[n as usize - 1]
        };
        let mut l = self.stream.launch_builder(f);
        l.arg(weights).arg(xp).arg(y).arg(&columns).arg(&rows);
        let cfg = grid(if kparallel { rows } else { rows.div_ceil(4) }, 128);
        // SAFETY: validated Q4_0 size, x holds n*columns, y n*rows floats.
        unsafe { l.launch(cfg) }.map_err(|e| failure("Matvec lot", e))?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tiled_gate(
        &self,
        codes: &CudaSlice<u16>,
        scales: &CudaSlice<u16>,
        xp: &CudaSlice<f32>,
        g: &mut CudaSlice<f32>,
        u: &mut CudaSlice<f32>,
        columns: u32,
        intermediate: u32,
        n: u32,
    ) -> Result<(), DiagnosticError> {
        check(n, g.len().min(u.len()), (n * intermediate) as usize)?;
        let tiles = ((intermediate * 2).div_ceil(4) * (columns / 32)) as usize;
        if xp.len() < (n * columns) as usize
            || codes.len() != tiles * 32
            || scales.len() != tiles * 4
        {
            return Err(DiagnosticError("Gate lot invalide".into()));
        }
        let mut l = self.stream.launch_builder(&self.tiled_gate[n as usize - 1]);
        l.arg(codes)
            .arg(scales)
            .arg(xp)
            .arg(g)
            .arg(u)
            .arg(&columns)
            .arg(&intermediate);
        // SAFETY: tile buffers sized for 2*intermediate rows, outputs n*intermediate.
        unsafe { l.launch(grid((intermediate * 2).div_ceil(4), 128)) }
            .map_err(|e| failure("Gate lot", e))?;
        Ok(())
    }
    pub(crate) fn rms(
        &self,
        x: &CudaSlice<f32>,
        scales: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        width: u32,
        eps: f32,
        n: u32,
    ) -> Result<(), DiagnosticError> {
        check(n, x.len().min(y.len()), (n * width) as usize)?;
        let mut l = self.stream.launch_builder(&self.rms);
        l.arg(x).arg(scales).arg(y).arg(&width).arg(&eps);
        // SAFETY: n vectors of width in x and y, one block each.
        unsafe { l.launch(grid(n, 256)) }.map_err(|e| failure("Norme lot", e))?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rms_add_rms(
        &self,
        projected: &CudaSlice<f32>,
        residual: &CudaSlice<f32>,
        s1: &CudaSlice<f32>,
        s2: &CudaSlice<f32>,
        h: &mut CudaSlice<f32>,
        normalized: &mut CudaSlice<f32>,
        width: u32,
        eps1: f32,
        eps2: f32,
        n: u32,
    ) -> Result<(), DiagnosticError> {
        check(n, h.len().min(normalized.len()), (n * width) as usize)?;
        if width > 4096 {
            return Err(DiagnosticError("Largeur lot > 4096".into()));
        }
        let mut l = self.stream.launch_builder(&self.rms_add_rms);
        l.arg(projected)
            .arg(residual)
            .arg(s1)
            .arg(s2)
            .arg(h)
            .arg(normalized)
            .arg(&width)
            .arg(&eps1)
            .arg(&eps2);
        // SAFETY: n vectors of width <= 4096 everywhere, one block each.
        unsafe { l.launch(grid(n, 256)) }.map_err(|e| failure("Norme/résiduel lot", e))?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rms_residual_scale(
        &self,
        down: &CudaSlice<f32>,
        residual: &CudaSlice<f32>,
        s: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        width: u32,
        eps: f32,
        scale: f32,
        n: u32,
    ) -> Result<(), DiagnosticError> {
        check(n, out.len(), (n * width) as usize)?;
        if width > 4096 {
            return Err(DiagnosticError("Largeur lot > 4096".into()));
        }
        let mut l = self.stream.launch_builder(&self.rms_residual_scale);
        l.arg(down)
            .arg(residual)
            .arg(s)
            .arg(out)
            .arg(&width)
            .arg(&eps)
            .arg(&scale);
        // SAFETY: n vectors of width <= 4096, one block each.
        unsafe { l.launch(grid(n, 256)) }.map_err(|e| failure("Post-norme lot", e))?;
        Ok(())
    }
    pub(crate) fn gelu(
        &self,
        g: &CudaSlice<f32>,
        u: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        total: u32,
    ) -> Result<(), DiagnosticError> {
        if g.len().min(u.len()).min(out.len()) < total as usize {
            return Err(DiagnosticError("GELU lot invalide".into()));
        }
        let mut l = self.stream.launch_builder(&self.gelu);
        l.arg(g).arg(u).arg(out).arg(&total);
        // SAFETY: total elements in all three buffers, bounded threads.
        unsafe { l.launch(grid(total.div_ceil(256), 256)) }
            .map_err(|e| failure("GELU lot", e))?;
        Ok(())
    }
}

/// Scratch shared by all layers (they run one after another on one stream).
pub(crate) struct BatchBuffers {
    pub input: CudaSlice<f32>,
    pub output: CudaSlice<f32>,
    pub normalized: CudaSlice<f32>,
    pub permuted: CudaSlice<f32>,
    pub raw: [CudaSlice<f32>; 3],
    pub attention: CudaSlice<f32>,
    /// Rotated queries of the batch, [n][q_heads*width].
    pub queries: CudaSlice<f32>,
    /// Attention scores of the batch, [n][q_heads][MAX_CONTEXT].
    pub scores: CudaSlice<f32>,
    pub projected: CudaSlice<f32>,
    pub hidden: CudaSlice<f32>,
    pub gate: CudaSlice<f32>,
    pub up: CudaSlice<f32>,
    pub active: CudaSlice<f32>,
    pub down: CudaSlice<f32>,
}
impl BatchBuffers {
    /// `qkv_rows` and `attention_width` are maxima over layers.
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        width: u32,
        qkv_rows: [u32; 3],
        attention_width: u32,
        intermediate: u32,
        q_heads: u32,
    ) -> Result<Self, DiagnosticError> {
        let alloc = |n: u32| {
            stream
                .alloc_zeros::<f32>(MAX_BATCH * n as usize)
                .map_err(|e| failure("Buffer lot", e))
        };
        let widest = width.max(intermediate).max(attention_width);
        Ok(Self {
            input: alloc(width)?,
            output: alloc(width)?,
            normalized: alloc(width)?,
            permuted: alloc(widest)?,
            raw: [alloc(qkv_rows[0])?, alloc(qkv_rows[1])?, alloc(qkv_rows[2])?],
            attention: alloc(attention_width)?,
            queries: alloc(attention_width)?,
            scores: stream
                .alloc_zeros::<f32>(MAX_BATCH * q_heads as usize * super::stack::MAX_CONTEXT as usize)
                .map_err(|e| failure("Scores lot", e))?,
            projected: alloc(width)?,
            hidden: alloc(width)?,
            gate: alloc(intermediate)?,
            up: alloc(intermediate)?,
            active: alloc(intermediate)?,
            down: alloc(width)?,
        })
    }
}
