use super::failure;
use adaforge_application::DiagnosticError;
use adaforge_domain::Ffn;
use cudarc::{
    driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::sync::Arc;
pub(crate) struct FfnKernel {
    stream: Arc<CudaStream>,
    project: CudaFunction,
    q8: Option<Q8Ffn>,
    tiled_gate: Option<(CudaFunction, CudaSlice<u16>, CudaSlice<u16>)>,
    gate_up: CudaFunction,
    rms: CudaFunction,
    gelu: CudaFunction,
    fused_gelu: Option<CudaFunction>,
    residual: CudaFunction,
    native_gate: Option<[CudaSlice<u8>; 2]>,
    down_weights: CudaSlice<u8>,
    pre_scale: CudaSlice<f32>,
    post_scale: CudaSlice<f32>,
    normalized: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    active: CudaSlice<f32>,
    down: CudaSlice<f32>,
    post: CudaSlice<f32>,
    output: CudaSlice<f32>,
    width: u32,
    intermediate: u32,
    pre_eps: f32,
    post_eps: f32,
    layer_scale: f32,
    fused_post: CudaFunction,
}
/// Fused norm/residual kernels (bit-identical). `ADAFORGE_FUSED_RMS=0` disables them.
pub(crate) fn fused_rms() -> bool {
    std::env::var("ADAFORGE_FUSED_RMS").as_deref() != Ok("0")
}
impl FfnKernel {
    pub(crate) fn strategy_name(&self) -> &'static str {
        if self.q8.is_some() {
            "q8"
        } else if self.tiled_gate.is_some() {
            "q4_tiled"
        } else {
            "q4_native"
        }
    }
    pub(crate) fn matrix_weight_bytes(&self) -> usize {
        self.down_weights.len()
            + self
                .native_gate
                .as_ref()
                .map_or(0, |w| w[0].len() + w[1].len())
            + self.tiled_gate.as_ref().map_or(0, |(_, codes, scales)| {
                (codes.len() + scales.len()) * size_of::<u16>()
            })
    }
    pub(crate) fn new(stream: Arc<CudaStream>, ffn: &Ffn) -> Result<Self, DiagnosticError> {
        let experimental_q8 = std::env::var("ADAFORGE_EXPERIMENTAL_Q8").as_deref() == Ok("1");
        Self::with_precision(stream, ffn, experimental_q8)
    }
    fn with_precision(
        stream: Arc<CudaStream>,
        ffn: &Ffn,
        experimental_q8: bool,
    ) -> Result<Self, DiagnosticError> {
        let source = format!(
            "{}\n{}\n{}\n{}\n{}",
            include_str!("../../../kernels/ada/q4_matvec.cu"),
            include_str!("../../../kernels/ada/rms_norm.cu"),
            include_str!("../../../kernels/ada/ffn.cu"),
            include_str!("../../../kernels/ada/q4_tiled_probe.cu"),
            if experimental_q8 {
                include_str!("../../../kernels/ada/q4_q8.cu")
            } else {
                ""
            }
        );
        let source = if std::env::var("ADAFORGE_EXPERIMENTAL_STREAM_WEIGHTS").as_deref() == Ok("1")
        {
            format!("#define ADAFORGE_STREAM_WEIGHTS\n{source}")
        } else {
            source
        };
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| failure("Compilation FFN", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| failure("Module FFN", e))?;
        let load = |name| {
            module
                .load_function(name)
                .map_err(|e| failure("Kernel FFN", e))
        };
        // The long down projection shares each row across four warps.
        let project = load("q4_matvec_kparallel")?;
        let gate_up = load("q4_gate_up")?;
        let rms = load("rms_norm")?;
        let gelu = load("gelu_gate")?;
        let residual = load("ffn_residual_scale")?;
        let fused_post = load("rms_residual_scale")?;
        // Qualified only for Gemma's fixed dense FFN geometry. Explicit 0 keeps
        // the native path available for regression checks; 1 forces the prototype.
        let use_tiled = std::env::var("ADAFORGE_EXPERIMENTAL_TILED_GATE")
            .map(|value| value == "1")
            .unwrap_or(ffn.width() == 3840 && ffn.intermediate() == 15360);
        let tiled_gate = if !experimental_q8 && use_tiled {
            let matrices = ffn.matrices();
            let rows = matrices[0].rows() as usize;
            let blocks = matrices[0].columns() as usize / 32;
            let tiles = (rows * 2).div_ceil(4) * blocks;
            let mut codes = vec![0u16; tiles * 32];
            let mut scales = vec![0u16; tiles * 4];
            for row in 0..rows * 2 {
                let bytes = matrices[row / rows].bytes();
                for b in 0..blocks {
                    let p = ((row % rows) * blocks + b) * 18;
                    let tile = (row / 4) * blocks + b;
                    scales[tile * 4 + row % 4] = u16::from_le_bytes([bytes[p], bytes[p + 1]]);
                    for word in 0..8 {
                        let offset = p + 2 + word * 2;
                        codes[(tile * 8 + word) * 4 + row % 4] =
                            u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                    }
                }
            }
            Some((
                load("q4_tiled_gate")?,
                stream
                    .clone_htod(&codes)
                    .map_err(|e| failure("Tiled codes", e))?,
                stream
                    .clone_htod(&scales)
                    .map_err(|e| failure("Tiled scales", e))?,
            ))
        } else {
            None
        };
        let matrices = ffn.matrices();
        let upload = |index: usize| {
            stream
                .clone_htod(matrices[index].bytes())
                .map_err(|e| failure("Poids FFN", e))
        };
        // A tiled session owns only the packed gate/up representation.
        let fused_gelu = if tiled_gate.is_some()
            && ffn.intermediate().is_multiple_of(4)
            && std::env::var("ADAFORGE_EXPERIMENTAL_FUSED_GELU").as_deref() == Ok("1")
        {
            Some(load("q4_tiled_gate_gelu")?)
        } else {
            None
        };
        let native_gate = if tiled_gate.is_none() {
            Some([upload(0)?, upload(1)?])
        } else {
            None
        };
        let down_weights = upload(2)?;
        let pre_scale = stream
            .clone_htod(ffn.pre().weights())
            .map_err(|e| failure("Échelles pré-FFN", e))?;
        let post_scale = stream
            .clone_htod(ffn.post().weights())
            .map_err(|e| failure("Échelles post-FFN", e))?;
        let alloc = |n| {
            stream
                .alloc_zeros::<f32>(n as usize)
                .map_err(|e| failure("Buffer FFN", e))
        };
        let width = ffn.width();
        let intermediate = ffn.intermediate();
        let normalized = alloc(width)?;
        let gate = alloc(intermediate)?;
        let up = alloc(intermediate)?;
        let active = alloc(intermediate)?;
        let down = alloc(width)?;
        let post = alloc(width)?;
        let output = alloc(width)?;
        let q8 = if experimental_q8 {
            Some(Q8Ffn {
                quant: load("quantize_q8")?,
                gate_up: load("q4_q8_gate_up")?,
                down: load("q4_q8_matvec")?,
                values: stream
                    .alloc_zeros(width.max(intermediate) as usize)
                    .map_err(|e| failure("Q8 values", e))?,
                scales: stream
                    .alloc_zeros((width.max(intermediate) / 32) as usize)
                    .map_err(|e| failure("Q8 scales", e))?,
            })
        } else {
            None
        };
        Ok(Self {
            stream,
            project,
            q8,
            tiled_gate,
            gate_up,
            rms,
            gelu,
            fused_gelu,
            residual,
            native_gate,
            down_weights,
            pre_scale,
            post_scale,
            normalized,
            gate,
            up,
            active,
            down,
            post,
            output,
            width,
            intermediate,
            pre_eps: ffn.pre().epsilon(),
            post_eps: ffn.post().epsilon(),
            fused_post,
            layer_scale: ffn.layer_scale(),
        })
    }
    /// Pre-norm scales, epsilon and target buffer, for a producer that fuses the norm.
    pub(crate) fn pre_norm_parts(&mut self) -> (&CudaSlice<f32>, f32, &mut CudaSlice<f32>) {
        (&self.pre_scale, self.pre_eps, &mut self.normalized)
    }
    pub(crate) fn run(&mut self, input: &CudaSlice<f32>) -> Result<(), DiagnosticError> {
        self.run_inner(input, false)
    }
    /// `normalized` was already written by the producer (see `pre_norm_parts`).
    pub(crate) fn run_prenormed(&mut self, input: &CudaSlice<f32>) -> Result<(), DiagnosticError> {
        self.run_inner(input, true)
    }
    fn run_inner(&mut self, input: &CudaSlice<f32>, prenormed: bool) -> Result<(), DiagnosticError> {
        if input.len() != self.width as usize || input.context() != self.stream.context() {
            return Err(DiagnosticError("Entrée FFN incompatible".into()));
        }
        if !prenormed {
            norm(
                &self.stream,
                &self.rms,
                input,
                &self.pre_scale,
                &mut self.normalized,
                self.width,
                self.pre_eps,
            )?;
        }
        if let Some(q8) = self.q8.as_mut() {
            let native = self
                .native_gate
                .as_ref()
                .ok_or_else(|| DiagnosticError("Poids natifs Q8 absents".into()))?;
            q8.quantize(&self.stream, &self.normalized, self.width)?;
            let mut launch = self.stream.launch_builder(&q8.gate_up);
            launch
                .arg(&native[0])
                .arg(&native[1])
                .arg(&q8.values)
                .arg(&q8.scales)
                .arg(&mut self.gate)
                .arg(&mut self.up)
                .arg(&self.width)
                .arg(&self.intermediate);
            // SAFETY: Ffn validates identical matrix shapes; Q8 buffers cover width.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: ((self.intermediate * 2).div_ceil(4), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| failure("Experimental Q8 gate/up", e))?;
        } else if let Some((function, codes, scales)) = self.tiled_gate.as_ref() {
            if let Some(fused) = self.fused_gelu.as_ref() {
                let mut launch = self.stream.launch_builder(fused);
                launch
                    .arg(codes)
                    .arg(scales)
                    .arg(&self.normalized)
                    .arg(&mut self.active)
                    .arg(&self.width)
                    .arg(&self.intermediate);
                // SAFETY: tiled rows are a multiple of four; two groups each cover four rows.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (self.intermediate / 4, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| failure("Fused gate/up GELU", e))?;
            } else {
                let mut launch = self.stream.launch_builder(function);
                launch
                    .arg(codes)
                    .arg(scales)
                    .arg(&self.normalized)
                    .arg(&mut self.gate)
                    .arg(&mut self.up)
                    .arg(&self.width)
                    .arg(&self.intermediate);
                // SAFETY: padded lossless tiles cover gate then up; four warps per CTA
                // cooperate on four rows, with individually bounded stores.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: ((self.intermediate * 2).div_ceil(4), 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| failure("Tiled gate/up", e))?;
            }
        } else {
            let native = self
                .native_gate
                .as_ref()
                .ok_or_else(|| DiagnosticError("Poids gate/up natifs absents".into()))?;
            let mut launch = self.stream.launch_builder(&self.gate_up);
            launch
                .arg(&native[0])
                .arg(&native[1])
                .arg(&self.normalized)
                .arg(&mut self.gate)
                .arg(&mut self.up)
                .arg(&self.width)
                .arg(&self.intermediate);
            // SAFETY: Ffn validates both matrices and their identical dimensions; outputs are distinct.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: ((self.intermediate * 2).div_ceil(4), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| failure("Gate/up fusionnés", e))?;
        }
        if self.fused_gelu.is_none() {
            let config = LaunchConfig {
                grid_dim: (self.intermediate.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream.launch_builder(&self.gelu);
            launch
                .arg(&self.gate)
                .arg(&self.up)
                .arg(&mut self.active)
                .arg(&self.intermediate);
            // SAFETY: trois buffers intermediate, threads bornés par n, sorties distinctes,
            // opérations précédentes et suivantes sur le même stream.
            unsafe { launch.launch(config) }.map_err(|e| failure("GELU gate/up", e))?;
        }
        if let Some(q8) = self.q8.as_mut() {
            q8.quantize(&self.stream, &self.active, self.intermediate)?;
            let mut launch = self.stream.launch_builder(&q8.down);
            launch
                .arg(&self.down_weights)
                .arg(&q8.values)
                .arg(&q8.scales)
                .arg(&mut self.down)
                .arg(&self.intermediate)
                .arg(&self.width);
            // SAFETY: validated down matrix, four complete row warps per 128-thread block.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (self.width.div_ceil(4), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| failure("Experimental Q8 down", e))?;
        } else {
            down_matvec(
                &self.stream,
                &self.project,
                &self.down_weights,
                &self.active,
                &mut self.down,
                self.intermediate,
                self.width,
            )?;
        }
        if fused_rms() && self.width <= 4096 {
            let mut launch = self.stream.launch_builder(&self.fused_post);
            launch
                .arg(&self.down)
                .arg(input)
                .arg(&self.post_scale)
                .arg(&mut self.output)
                .arg(&self.width)
                .arg(&self.post_eps)
                .arg(&self.layer_scale);
            // SAFETY: width <= 4096 vectors, one 256-thread block, output distinct.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| failure("Post-norme FFN fusionnée", e))?;
            return Ok(());
        }
        norm(
            &self.stream,
            &self.rms,
            &self.down,
            &self.post_scale,
            &mut self.post,
            self.width,
            self.post_eps,
        )?;
        let config = LaunchConfig {
            grid_dim: (self.width.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.residual);
        launch
            .arg(&self.post)
            .arg(input)
            .arg(&mut self.output)
            .arg(&self.width)
            .arg(&self.layer_scale);
        // SAFETY: trois buffers width, entrée résiduelle non normalisée, échelle finie,
        // kernel borné et sortie séparée. Aucun transfert hôte entre ces étapes.
        unsafe { launch.launch(config) }.map_err(|e| failure("Résiduel FFN", e))?;
        Ok(())
    }
    pub(crate) fn output(&self) -> &CudaSlice<f32> {
        &self.output
    }
    pub(crate) fn snapshot(&self) -> Result<Vec<f32>, DiagnosticError> {
        let out = self
            .stream
            .clone_dtoh(&self.output)
            .map_err(|e| failure("Retour FFN", e))?;
        self.stream
            .synchronize()
            .map_err(|e| failure("Synchronisation FFN", e))?;
        Ok(out)
    }
}
fn down_matvec(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    weights: &CudaSlice<u8>,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    columns: u32,
    rows: u32,
) -> Result<(), DiagnosticError> {
    let config = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(function);
    launch
        .arg(weights)
        .arg(input)
        .arg(output)
        .arg(&columns)
        .arg(&rows);
    // SAFETY: appels internes avec matrices et tailles validées par Ffn, bloc 128,
    // kernel K-parallel : un bloc entier par ligne, les quatre warps atteignent
    // la barriere ; buffers de bonne taille, sortie exclusive et ressources vivantes.
    unsafe { launch.launch(config) }.map_err(|e| failure("Matvec FFN", e))?;
    Ok(())
}
fn norm(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    input: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    width: u32,
    epsilon: f32,
) -> Result<(), DiagnosticError> {
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(function);
    launch
        .arg(input)
        .arg(weights)
        .arg(output)
        .arg(&width)
        .arg(&epsilon);
    // SAFETY: appels internes, trois buffers width, bloc 256, epsilon validé,
    // sortie distincte ; même stream CUDA que les autres opérations.
    unsafe { launch.launch(config) }.map_err(|e| failure("RMSNorm FFN", e))?;
    Ok(())
}
struct Q8Ffn {
    quant: CudaFunction,
    gate_up: CudaFunction,
    down: CudaFunction,
    values: CudaSlice<i8>,
    scales: CudaSlice<f32>,
}
impl Q8Ffn {
    fn quantize(
        &mut self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        columns: u32,
    ) -> Result<(), DiagnosticError> {
        let mut launch = stream.launch_builder(&self.quant);
        launch
            .arg(input)
            .arg(&mut self.values)
            .arg(&mut self.scales)
            .arg(&columns);
        // SAFETY: Ffn widths are multiples of 32 and scratch covers the larger width.
        // This launch and the consumer share the same private stream, also during graph capture.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (columns.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Experimental activation quantization", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adaforge_domain::{Q4Matrix, RmsNorm};
    use cudarc::driver::CudaContext;
    fn matrix(columns: u32, rows: u32, seed: u32) -> Q4Matrix {
        let mut b = Vec::new();
        for i in 0..columns / 32 * rows {
            b.extend(0x2800_u16.to_le_bytes());
            b.extend(
                (0..16)
                    .map(|j| ((i + j + seed) % 16) as u8 | (((i + 15 - j + seed) % 16) as u8) << 4),
            );
        }
        Q4Matrix::new(columns, rows, b).unwrap()
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_dense_ffn_matches_cpu_and_keeps_residual() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let f = Ffn::new(
            RmsNorm::new(vec![1.25; 32], 1e-6).unwrap(),
            matrix(32, 64, 0),
            matrix(32, 64, 2),
            matrix(64, 32, 3),
            RmsNorm::new(vec![0.75; 32], 1e-6).unwrap(),
            0.5,
        )
        .unwrap();
        let mut kernel = FfnKernel::new(stream.clone(), &f).unwrap();
        let native_bytes = kernel
            .native_gate
            .as_ref()
            .map_or(0, |w| w[0].len() + w[1].len());
        let tiled_bytes = kernel.tiled_gate.as_ref().map_or(0, |(_, codes, scales)| {
            (codes.len() + scales.len()) * size_of::<u16>()
        });
        // Regression guard: packing must replace, not duplicate, GPU gate/up weights.
        assert_eq!(
            native_bytes + tiled_bytes,
            f.matrices()[0].bytes().len() + f.matrices()[1].bytes().len()
        );
        for zero in [true, false] {
            let input: Vec<f32> = (0..32)
                .map(|i| {
                    if zero {
                        0.0
                    } else {
                        (i % 11) as f32 * 0.125 - 0.5
                    }
                })
                .collect();
            let reference = f.reference(&input).unwrap();
            let device = stream.clone_htod(&input).unwrap();
            kernel.run(&device).unwrap();
            super::super::matvec::compare(&reference, &kernel.snapshot().unwrap()).unwrap();
        }
    }
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn experimental_q8_ffn_matches_quantized_cpu_oracle() {
        fn quantize(input: &[f32]) -> Vec<f32> {
            input
                .chunks_exact(32)
                .flat_map(|block| {
                    let maximum = block.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    let d = maximum / 127.0;
                    block.iter().map(move |&v| {
                        if d == 0.0 {
                            0.0
                        } else {
                            (v / d).round_ties_even().clamp(-127.0, 127.0) * d
                        }
                    })
                })
                .collect()
        }
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let f = Ffn::new(
            RmsNorm::new(vec![1.25; 32], 1e-6).unwrap(),
            matrix(32, 64, 0),
            matrix(32, 64, 2),
            matrix(64, 32, 3),
            RmsNorm::new(vec![0.75; 32], 1e-6).unwrap(),
            0.5,
        )
        .unwrap();
        let mut kernel = FfnKernel::with_precision(stream.clone(), &f, true).unwrap();
        for zero in [true, false] {
            let input: Vec<f32> = (0..32)
                .map(|i| {
                    if zero {
                        0.0
                    } else {
                        (i % 11) as f32 * 0.125 - 0.5
                    }
                })
                .collect();
            let normalized: Vec<f32> = f
                .pre()
                .reference(&input)
                .unwrap()
                .into_iter()
                .map(|x| x as f32)
                .collect();
            let normalized = quantize(&normalized);
            let [g, u, d] = f.matrices();
            let gate = g.reference_matvec(&normalized).unwrap();
            let up = u.reference_matvec(&normalized).unwrap();
            let active: Vec<f32> = gate
                .iter()
                .zip(up)
                .map(|(&g, u)| {
                    let g = f64::from(g as f32);
                    let activation = (0.5
                        * g
                        * (1.0 + (0.7978845608028654 * (g + 0.044715 * g * g * g)).tanh()))
                        as f32;
                    activation * (u as f32)
                })
                .collect();
            let down: Vec<f32> = d
                .reference_matvec(&quantize(&active))
                .unwrap()
                .into_iter()
                .map(|v| v as f32)
                .collect();
            let post = f.post().reference(&down).unwrap();
            let expected: Vec<f64> = post
                .iter()
                .zip(&input)
                .map(|(&p, &x)| f64::from((p as f32 + x) * f.layer_scale()))
                .collect();
            let device = stream.clone_htod(&input).unwrap();
            kernel.run(&device).unwrap();
            super::super::matvec::compare(&expected, &kernel.snapshot().unwrap()).unwrap();
        }
    }
}
