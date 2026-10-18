use super::failure;
use adaforge_application::DiagnosticError;
use cudarc::{
    driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::sync::Arc;
/// Most rows `select_many` handles in one call.
pub(crate) const MANY: usize = 16;
pub(crate) struct GpuGreedy {
    stream: Arc<CudaStream>,
    partial: CudaFunction,
    finish: CudaFunction,
    values: CudaSlice<f32>,
    ids: CudaSlice<u32>,
    bad: CudaSlice<u32>,
    result: CudaSlice<u32>,
    /// Results of `select_many`, two u32 per row (id, bad flag).
    results: CudaSlice<u32>,
    count: u32,
    groups: u32,
}
impl GpuGreedy {
    pub(crate) fn new(stream: Arc<CudaStream>, count: u32) -> Result<Self, DiagnosticError> {
        if count == 0 || count > 1000000 {
            return Err(DiagnosticError("Vocabulaire greedy invalide".into()));
        }
        let ptx = compile_ptx_with_opts(
            include_str!("../../../kernels/ada/greedy.cu"),
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| failure("Compilation greedy GPU", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| failure("Module greedy GPU", e))?;
        let partial = module
            .load_function("greedy_partial")
            .map_err(|e| failure("Greedy partiel", e))?;
        let finish = module
            .load_function("greedy_final")
            .map_err(|e| failure("Greedy final", e))?;
        let groups = count.div_ceil(256);
        Ok(Self {
            partial,
            finish,
            count,
            groups,
            values: stream
                .alloc_zeros(groups as usize)
                .map_err(|e| failure("Valeurs greedy", e))?,
            ids: stream
                .alloc_zeros(groups as usize)
                .map_err(|e| failure("IDs greedy", e))?,
            bad: stream
                .alloc_zeros(groups as usize)
                .map_err(|e| failure("Validité greedy", e))?,
            result: stream
                .alloc_zeros(2)
                .map_err(|e| failure("Résultat greedy", e))?,
            results: stream
                .alloc_zeros(2 * MANY)
                .map_err(|e| failure("Résultats greedy", e))?,
            stream,
        })
    }
    /// Greedy ids of `n` consecutive rows of `count` logits, with a single
    /// device-to-host read. Each id equals `select` on that row.
    pub(crate) fn select_many(
        &mut self,
        input: &CudaSlice<f32>,
        n: usize,
    ) -> Result<Vec<u32>, DiagnosticError> {
        let count = self.count as usize;
        if n == 0 || n > MANY || input.len() < n * count || input.context() != self.stream.context() {
            return Err(DiagnosticError("Entrée greedy multiple incompatible".into()));
        }
        for row in 0..n {
            let x = input.slice(row * count..(row + 1) * count);
            let mut launch = self.stream.launch_builder(&self.partial);
            launch
                .arg(&x)
                .arg(&mut self.values)
                .arg(&mut self.ids)
                .arg(&mut self.bad)
                .arg(&self.count);
            // SAFETY: the view holds count logits; scratch sized for ceil(count/256) groups.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (self.groups, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| failure("Réduction greedy partielle", e))?;
            let mut out = self.results.slice_mut(2 * row..2 * row + 2);
            let mut launch = self.stream.launch_builder(&self.finish);
            launch
                .arg(&self.values)
                .arg(&self.ids)
                .arg(&self.bad)
                .arg(&mut out)
                .arg(&self.groups);
            // SAFETY: single block; scratch is consumed in stream order before reuse.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| failure("Réduction greedy finale", e))?;
        }
        let results = self
            .stream
            .clone_dtoh(&self.results.slice(0..2 * n))
            .map_err(|e| failure("Lecture IDs greedy", e))?;
        self.stream
            .synchronize()
            .map_err(|e| failure("Synchronisation greedy", e))?;
        results
            .chunks(2)
            .map(|r| {
                if r[1] != 0 || r[0] >= self.count {
                    Err(DiagnosticError("Logits non finis ou ID greedy invalide".into()))
                } else {
                    Ok(r[0])
                }
            })
            .collect()
    }
    pub(crate) fn select(&mut self, input: &CudaSlice<f32>) -> Result<u32, DiagnosticError> {
        if input.len() != self.count as usize || input.context() != self.stream.context() {
            return Err(DiagnosticError("Entrée greedy incompatible".into()));
        }
        let mut launch = self.stream.launch_builder(&self.partial);
        launch
            .arg(input)
            .arg(&mut self.values)
            .arg(&mut self.ids)
            .arg(&mut self.bad)
            .arg(&self.count);
        // SAFETY: count checked, groups ceil(count/256), shared arrays sized for 256 threads.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.groups, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Réduction greedy partielle", e))?;
        let mut launch = self.stream.launch_builder(&self.finish);
        launch
            .arg(&self.values)
            .arg(&self.ids)
            .arg(&self.bad)
            .arg(&mut self.result)
            .arg(&self.groups);
        // SAFETY: single 256-thread block, validated scratch buffers and two-u32 result.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Réduction greedy finale", e))?;
        let result = self
            .stream
            .clone_dtoh(&self.result)
            .map_err(|e| failure("Lecture ID greedy", e))?;
        self.stream
            .synchronize()
            .map_err(|e| failure("Synchronisation greedy", e))?;
        if result[1] != 0 || result[0] >= self.count {
            return Err(DiagnosticError(
                "Logits non finis ou ID greedy invalide".into(),
            ));
        }
        Ok(result[0])
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_argmax_matches_cpu_ties_boundaries_and_rejects_nonfinite() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        for size in [1, 5, 255, 256, 257, 1025, 262144] {
            let mut select = GpuGreedy::new(stream.clone(), size).unwrap();
            let mut values = vec![-3.0; size as usize];
            values[size as usize - 1] = 1.0;
            values[0] = 1.0;
            let device = stream.clone_htod(&values).unwrap();
            assert_eq!(
                select.select(&device).unwrap(),
                adaforge_domain::greedy(&values).unwrap()
            );
            values[0] = -7.0;
            let device = stream.clone_htod(&values).unwrap();
            assert_eq!(
                select.select(&device).unwrap(),
                adaforge_domain::greedy(&values).unwrap()
            );
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                values[0] = invalid;
                let device = stream.clone_htod(&values).unwrap();
                assert!(select.select(&device).is_err());
            }
        }
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn select_many_matches_select_per_row() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        for size in [1u32, 257, 262144] {
            let mut select = GpuGreedy::new(stream.clone(), size).unwrap();
            let n = 5usize;
            let mut values = vec![-3.0f32; n * size as usize];
            for row in 0..n {
                let base = row * size as usize;
                values[base + (row * 7919) % size as usize] = 2.0;
                values[base] = 2.0; // tie with the first index
            }
            let device = stream.clone_htod(&values).unwrap();
            let many = select.select_many(&device, n).unwrap();
            for row in 0..n {
                let one = stream
                    .clone_htod(&values[row * size as usize..(row + 1) * size as usize])
                    .unwrap();
                assert_eq!(many[row], select.select(&one).unwrap(), "size {size} row {row}");
            }
            values[size as usize] = f32::NAN;
            let device = stream.clone_htod(&values).unwrap();
            assert!(select.select_many(&device, n).is_err());
        }
    }
}
