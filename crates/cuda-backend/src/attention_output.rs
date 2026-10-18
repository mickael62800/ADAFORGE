use super::failure;
use adaforge_application::DiagnosticError;
use adaforge_domain::AttentionOutput;
use cudarc::{
    driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
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
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| failure("Compilation sortie attention", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| failure("Module sortie", e))?;
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
