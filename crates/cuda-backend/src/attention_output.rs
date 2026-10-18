use super::failure;
use adaforge_application::DiagnosticError;
use adaforge_domain::AttentionOutput;
use cudarc::{
    driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg}
};
use std::sync::Arc;

pub(crate) struct OutputKernel {
    stream: Arc<CudaStream>,
    project: CudaFunction,
    kparallel: bool,
    rms: CudaFunction,
    add: CudaFunction,
    weights: CudaSlice<u8>,
    scales: CudaSlice<f32>,
    projected: CudaSlice<f32>,
    normalized: CudaSlice<f32>,
    output: CudaSlice<f32>,
    fused: CudaFunction,
    columns: u32,
    rows: u32,
    epsilon: f32,
}
impl OutputKernel {
    pub(crate) fn kparallel_enabled(&self) -> bool {
        self.kparallel
    }
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        branch: &AttentionOutput,
    ) -> Result<Self, DiagnosticError> {
        let source = format!(
            "{}\n{}\n{}",
            include_str!("../../../kernels/ada/q4_matvec.cu"),
            include_str!("../../../kernels/ada/rms_norm.cu"),
            include_str!("../../../kernels/ada/vector_add.cu")
        );
        let module = super::cached_module(&stream, source, "Compilation sortie attention")?;
        let kparallel = matches!(
            std::env::var("ADAFORGE_EXPERIMENTAL_KPAR_PROJECTIONS").as_deref(),
            Ok("1" | "output")
        );
        let project = module
            .load_function(if kparallel {
                "q4_matvec_kparallel"
            } else {
                "q4_matvec"
            })
            .map_err(|e| failure("Projection sortie", e))?;
        let rms = module
            .load_function("rms_norm")
            .map_err(|e| failure("Norme sortie", e))?;
        let add = module
            .load_function("vector_add")
            .map_err(|e| failure("Résiduel sortie", e))?;
        let fused = module
            .load_function("rms_add_rms")
            .map_err(|e| failure("Norme fusionnée sortie", e))?;
        let columns = branch.matrix().columns();
        let rows = branch.matrix().rows();
        let weights = stream
            .clone_htod(branch.matrix().bytes())
            .map_err(|e| failure("Poids sortie", e))?;
        let scales = stream
            .clone_htod(branch.norm().weights())
            .map_err(|e| failure("Poids norme post-attention", e))?;
        let projected = stream
            .alloc_zeros::<f32>(rows as usize)
            .map_err(|e| failure("Buffer projeté", e))?;
        let normalized = stream
            .alloc_zeros::<f32>(rows as usize)
            .map_err(|e| failure("Buffer post-norme", e))?;
        let output = stream
            .alloc_zeros::<f32>(rows as usize)
            .map_err(|e| failure("Buffer résiduel", e))?;
        Ok(Self {
            stream,
            project,
            kparallel,
            rms,
            add,
            weights,
            scales,
            projected,
            normalized,
            output,
            fused,
            columns,
            rows,
            epsilon: branch.norm().epsilon(),
        })
    }
    pub(crate) fn run(
        &mut self,
        attention: &CudaSlice<f32>,
        residual: &CudaSlice<f32>,
    ) -> Result<(), DiagnosticError> {
        if attention.len() != self.columns as usize
            || residual.len() != self.rows as usize
            || attention.context() != self.stream.context()
            || residual.context() != self.stream.context()
        {
            return Err(DiagnosticError(
                "Dimensions/contexte sortie attention invalides".into(),
            ));
        }
        self.project(attention)?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.rms);
        launch
            .arg(&self.projected)
            .arg(&self.scales)
            .arg(&mut self.normalized)
            .arg(&self.rows)
            .arg(&self.epsilon);
        // SAFETY: un vecteur rows et rows échelles, bloc 256, sortie distincte.
        unsafe { launch.launch(config) }.map_err(|e| failure("Post-norme attention", e))?;
        let config = LaunchConfig {
            grid_dim: (self.rows.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.add);
        launch
            .arg(residual)
            .arg(&self.normalized)
            .arg(&mut self.output)
            .arg(&self.rows);
        // SAFETY: trois buffers rows, kernel borné par rows. Le résiduel est
        // l’entrée brute de la branche, pas son entrée normalisée. Aucun alias.
        unsafe { launch.launch(config) }.map_err(|e| failure("Ajout résiduel", e))?;
        Ok(())
    }
    fn project(&mut self, attention: &CudaSlice<f32>) -> Result<(), DiagnosticError> {
        let config = LaunchConfig {
            grid_dim: (
                if self.kparallel {
                    self.rows
                } else {
                    self.rows.div_ceil(4)
                },
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.project);
        launch
            .arg(&self.weights)
            .arg(attention)
            .arg(&mut self.projected)
            .arg(&self.columns)
            .arg(&self.rows);
        // SAFETY: matrice Q4_0 validée, entrée columns et sortie rows ; bloc 128,
        // ressources vivantes sur le même stream et buffers distincts.
        unsafe { launch.launch(config) }.map_err(|e| failure("Projection sortie", e))?;
        Ok(())
    }
    /// Fused path: projection, then one kernel for post-norm + residual add +
    /// the FFN pre-norm (written to `normalized`). Bit-identical to the
    /// separate kernels; requires rows <= 4096.
    pub(crate) fn run_fused(
        &mut self,
        attention: &CudaSlice<f32>,
        residual: &CudaSlice<f32>,
        next_scales: &CudaSlice<f32>,
        next_eps: f32,
        normalized: &mut CudaSlice<f32>,
    ) -> Result<(), DiagnosticError> {
        if attention.len() != self.columns as usize
            || residual.len() != self.rows as usize
            || next_scales.len() != self.rows as usize
            || normalized.len() != self.rows as usize
            || self.rows > 4096
        {
            return Err(DiagnosticError("Dimensions sortie fusionnée invalides".into()));
        }
        self.project(attention)?;
        let mut launch = self.stream.launch_builder(&self.fused);
        launch
            .arg(&self.projected)
            .arg(residual)
            .arg(&self.scales)
            .arg(next_scales)
            .arg(&mut self.output)
            .arg(normalized)
            .arg(&self.rows)
            .arg(&self.epsilon)
            .arg(&next_eps);
        // SAFETY: all vectors have rows <= 4096 elements (checked), one 256-thread
        // block, outputs distinct from inputs.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Norme/résiduel fusionnés", e))?;
        Ok(())
    }
    /// (weights, post-norm scales, post-norm epsilon, columns, rows) for batched passes.
    pub(crate) fn batch_parts(&self) -> (&CudaSlice<u8>, &CudaSlice<f32>, f32, u32, u32) {
        (&self.weights, &self.scales, self.epsilon, self.columns, self.rows)
    }
    pub(crate) fn output(&self) -> &CudaSlice<f32> {
        &self.output
    }
    pub(crate) fn snapshot(&self) -> Result<Vec<f32>, DiagnosticError> {
        let out = self
            .stream
            .clone_dtoh(&self.output)
            .map_err(|e| failure("Retour branche", e))?;
        self.stream
            .synchronize()
            .map_err(|e| failure("Synchronisation branche", e))?;
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use adaforge_domain::{Q4Matrix, RmsNorm};
    use cudarc::driver::CudaContext;
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_projects_normalizes_then_adds_original_residual() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut bytes = Vec::new();
        for row in 0..257 {
            bytes.extend(0x3800_u16.to_le_bytes());
            bytes.extend(
                (0..16).map(|j| ((row + j) % 16) as u8 | (((row + 15 - j) % 16) as u8) << 4),
            );
        }
        let branch = AttentionOutput::new(
            Q4Matrix::new(32, 257, bytes).unwrap(),
            RmsNorm::new((0..257).map(|i| (i % 7) as f32 - 3.0).collect(), 1e-6).unwrap(),
        )
        .unwrap();
        let mut kernel = OutputKernel::new(stream.clone(), &branch).unwrap();
        for zero in [true, false] {
            let a: Vec<f32> = (0..32)
                .map(|i| {
                    if zero {
                        0.0
                    } else {
                        (i % 11) as f32 * 0.125 - 0.5
                    }
                })
                .collect();
            let r: Vec<f32> = (0..257).map(|i| (i % 13) as f32 - 6.0).collect();
            let reference = branch.reference(&a, &r).unwrap();
            let ag = stream.clone_htod(&a).unwrap();
            let rg = stream.clone_htod(&r).unwrap();
            kernel.run(&ag, &rg).unwrap();
            super::super::matvec::compare(&reference, &kernel.snapshot().unwrap()).unwrap();
        }
    }
}

#[cfg(test)]
mod fused_tests {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn fused_norm_chains_match_separate_kernels_bits() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let source = format!(
            "{}
{}
{}",
            include_str!("../../../kernels/ada/rms_norm.cu"),
            include_str!("../../../kernels/ada/vector_add.cu"),
            include_str!("../../../kernels/ada/ffn.cu")
        );
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions { arch: Some("compute_89"), ..Default::default() },
        )
        .unwrap();
        let m = ctx.load_module(ptx).unwrap();
        let f = |n: &str| m.load_function(n).unwrap();
        let (rms, add, res, chain, post) =
            (f("rms_norm"), f("vector_add"), f("ffn_residual_scale"), f("rms_add_rms"), f("rms_residual_scale"));
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut state = 0x2545_f491u32;
        let mut next = |amp: f32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state % 65536) as f32 / 32768.0 - 1.0) * amp
        };
        for width in [1u32, 255, 256, 1000, 3840, 4096] {
            let n = width as usize;
            let mut vec = |amp: f32| stream.clone_htod(&(0..n).map(|_| next(amp)).collect::<Vec<_>>()).unwrap();
            let (p, r, s1, s2) = (vec(40.0), vec(3.0), vec(2.0), vec(2.0));
            let z = || stream.alloc_zeros::<f32>(n).unwrap();
            let (e1, e2, scale) = (1e-6f32, 1e-6f32, 0.73f32);
            let grid = LaunchConfig { grid_dim: (width.div_ceil(256), 1, 1), ..one };
            // Reference: separate kernels.
            let (mut a, mut h, mut nrm) = (z(), z(), z());
            let mut l = stream.launch_builder(&rms);
            l.arg(&p).arg(&s1).arg(&mut a).arg(&width).arg(&e1);
            // SAFETY: vectors of `width` <= 4096 elements, distinct outputs.
            unsafe { l.launch(one) }.unwrap();
            let mut l = stream.launch_builder(&add);
            l.arg(&r).arg(&a).arg(&mut h).arg(&width);
            // SAFETY: vectors of `width` <= 4096 elements, distinct outputs.
            unsafe { l.launch(grid) }.unwrap();
            let mut l = stream.launch_builder(&rms);
            l.arg(&h).arg(&s2).arg(&mut nrm).arg(&width).arg(&e2);
            // SAFETY: vectors of `width` <= 4096 elements, distinct outputs.
            unsafe { l.launch(one) }.unwrap();
            let (mut fh, mut fn_) = (z(), z());
            let mut l = stream.launch_builder(&chain);
            l.arg(&p).arg(&r).arg(&s1).arg(&s2).arg(&mut fh).arg(&mut fn_).arg(&width).arg(&e1).arg(&e2);
            // SAFETY: vectors of `width` <= 4096 elements, distinct outputs.
            unsafe { l.launch(one) }.unwrap();
            let (mut o, mut fo) = (z(), z());
            let mut l = stream.launch_builder(&res);
            l.arg(&a).arg(&r).arg(&mut o).arg(&width).arg(&scale);
            // SAFETY: vectors of `width` <= 4096 elements, distinct outputs.
            unsafe { l.launch(grid) }.unwrap();
            let mut l = stream.launch_builder(&post);
            l.arg(&p).arg(&r).arg(&s1).arg(&mut fo).arg(&width).arg(&e1).arg(&scale);
            // SAFETY: vectors of `width` <= 4096 elements, distinct outputs.
            unsafe { l.launch(one) }.unwrap();
            let get = |x: &cudarc::driver::CudaSlice<f32>| stream.clone_dtoh(x).unwrap();
            for (name, x, y) in [("h", &h, &fh), ("normalized", &nrm, &fn_), ("ffn out", &o, &fo)] {
                let (x, y) = (get(x), get(y));
                assert!(x.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()), "{name} width {width}");
            }
        }
    }
}
