use super::{failure, kv_cache::KvCache};
use adaforge_application::DiagnosticError;
use adaforge_domain::AttentionConfig;
use cudarc::{
    driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::sync::Arc;

pub(crate) struct AttentionKernel {
    parallel_exp: bool,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    dynamic: CudaFunction,
    output: CudaSlice<f32>,
    config: AttentionConfig,
}
impl AttentionKernel {
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        config: AttentionConfig,
    ) -> Result<Self, DiagnosticError> {
        let ptx = compile_ptx_with_opts(
            include_str!("../../../kernels/ada/attention.cu"),
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| failure("Compilation attention", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| failure("Module attention", e))?;
        let parallel_exp =
            std::env::var("ADAFORGE_EXPERIMENTAL_ATTENTION_EXP").as_deref() == Ok("1");
        let function = module
            .load_function(if parallel_exp {
                "causal_attention_parallel_exp"
            } else {
                "causal_attention"
            })
            .map_err(|e| failure("Kernel attention", e))?;
        let dynamic = module
            .load_function(if parallel_exp {
                "attention_dynamic_parallel_exp"
            } else {
                "attention_dynamic"
            })
            .map_err(|e| failure("Attention dynamique", e))?;
        let output = stream
            .alloc_zeros::<f32>(config.q_heads() as usize * config.width() as usize)
            .map_err(|e| failure("Sortie attention", e))?;
        Ok(Self {
            parallel_exp,
            stream,
            function,
            dynamic,
            output,
            config,
        })
    }
    pub(crate) fn run(
        &mut self,
        query: &CudaSlice<f32>,
        cache: &KvCache,
        position: u32,
    ) -> Result<(), DiagnosticError> {
        let super::kv_cache::ResidentKv {
            keys,
            values,
            tokens,
            layout,
        } = cache.resident()?;
        let c = self.config;
        if tokens > 4096
            || layout.heads() != c.kv_heads()
            || layout.width() != c.width()
            || query.len() != c.q_heads() as usize * c.width() as usize
            || query.context() != self.stream.context()
            || keys.context() != self.stream.context()
        {
            return Err(DiagnosticError(
                "Dimensions/contexte attention incompatibles ou limite de 4096 tokens dépassée"
                    .into(),
            ));
        }
        let range = c.visible_range(tokens, position).map_err(DiagnosticError)?;
        let width = c.width();
        let q_heads = c.q_heads();
        let kv_heads = c.kv_heads();
        let window = c.window().unwrap_or(0);
        let scale = c.scale();
        let config = LaunchConfig {
            grid_dim: (q_heads, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (range.len() * 4) as u32,
        };
        let mut launch = self.stream.launch_builder(&self.function);
        launch
            .arg(query)
            .arg(keys)
            .arg(values)
            .arg(&mut self.output)
            .arg(&width)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&tokens)
            .arg(&position)
            .arg(&window)
            .arg(&scale);
        // SAFETY: 256 threads, un bloc par tête ; formes et contexte vérifiés,
        // cache visible borné à tokens, scratch de visible.len f32. Buffers distincts
        // et vivants, même stream ; aucune écriture dans Q ou le cache.
        unsafe { launch.launch(config) }.map_err(|e| failure("Lancement attention", e))?;
        Ok(())
    }
    pub(crate) fn run_graph(
        &mut self,
        query: &CudaSlice<f32>,
        cache: &KvCache,
        state: &CudaSlice<u32>,
        capacity: u32,
    ) -> Result<(), DiagnosticError> {
        let data = cache.resident()?;
        let c = self.config;
        let width = c.width();
        let qheads = c.q_heads();
        let kvheads = c.kv_heads();
        let window = c.window().unwrap_or(0);
        let scale = c.scale();
        let mut launch = self.stream.launch_builder(&self.dynamic);
        launch
            .arg(query)
            .arg(data.keys)
            .arg(data.values)
            .arg(&mut self.output)
            .arg(&width)
            .arg(&qheads)
            .arg(&kvheads)
            .arg(&window)
            .arg(&scale)
            .arg(state);
        // SAFETY: private graph path, geometries validated by ResidentLayer; host bounds state position by capacity.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (qheads, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: capacity * 4,
            })
        }
        .map_err(|e| failure("Attention graphe", e))?;
        Ok(())
    }
    pub(crate) fn parallel_exp_enabled(&self) -> bool {
        self.parallel_exp
    }
    pub(crate) fn output(&self) -> &CudaSlice<f32> {
        &self.output
    }
    pub(crate) fn snapshot(&self) -> Result<Vec<f32>, DiagnosticError> {
        let out = self
            .stream
            .clone_dtoh(&self.output)
            .map_err(|e| failure("Retour attention", e))?;
        self.stream
            .synchronize()
            .map_err(|e| failure("Synchronisation attention", e))?;
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn parallel_exp_preserves_attention_bits_for_real_head_widths() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(
                compile_ptx_with_opts(
                    include_str!("../../../kernels/ada/attention.cu"),
                    CompileOptions {
                        arch: Some("compute_89"),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let qheads = 4u32;
        let kvheads = 2u32;
        let scale = 1.0f32;
        for width in [256u32, 512] {
            for tokens in [1u32, 3, 31, 63, 64, 65, 127, 512, 1025] {
                let q = stream
                    .clone_htod(
                        &(0..qheads * width)
                            .map(|i| (i % 17) as f32 * 0.1 - 0.8)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let k = stream
                    .clone_htod(
                        &(0..tokens * kvheads * width)
                            .map(|i| (i % 23) as f32 * 0.07 - 0.7)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let v = stream
                    .clone_htod(
                        &(0..tokens * kvheads * width)
                            .map(|i| (i % 19) as f32 * 0.09 - 0.5)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                for window in [0u32, 64] {
                    let position = tokens - 1;
                    let state = stream.clone_htod(&[0u32, position]).unwrap();
                    let mut outputs = Vec::new();
                    for name in [
                        "causal_attention",
                        "causal_attention_parallel_exp",
                        "attention_dynamic_parallel_exp",
                    ] {
                        let kernel = module.load_function(name).unwrap();
                        let mut out = stream
                            .alloc_zeros::<f32>((qheads * width) as usize)
                            .unwrap();
                        let mut launch = stream.launch_builder(&kernel);
                        launch
                            .arg(&q)
                            .arg(&k)
                            .arg(&v)
                            .arg(&mut out)
                            .arg(&width)
                            .arg(&qheads)
                            .arg(&kvheads);
                        if name.starts_with("attention_dynamic") {
                            launch.arg(&window).arg(&scale).arg(&state);
                        } else {
                            launch.arg(&tokens).arg(&position).arg(&window).arg(&scale);
                        }
                        // SAFETY: complete query/cache dimensions, one 256-thread block per Q head, scratch covers all visible tokens.
                        unsafe {
                            launch.launch(LaunchConfig {
                                grid_dim: (qheads, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: tokens * 4,
                            })
                        }
                        .unwrap();
                        let values = stream.clone_dtoh(&out).unwrap();
                        stream.synchronize().unwrap();
                        outputs.push(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    }
                    assert_eq!(
                        outputs[0], outputs[1],
                        "width={width}, tokens={tokens}, window={window}"
                    );
                    assert_eq!(outputs[0], outputs[2]);
                }
            }
        }
    }
    use super::*;
    use adaforge_domain::KvLayout;
    use cudarc::driver::CudaContext;
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_masks_future_and_expired_tokens_and_maps_groups() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        for (tokens, window, position) in [
            (5, None, 2),
            (5, Some(1), 2),
            (1025, Some(1024), 1024),
            (1025, None, 1024),
        ] {
            let config = AttentionConfig::new(4, 2, 3, window, 1.0).unwrap();
            let mut keys = vec![0.0; tokens as usize * 6];
            let mut values = vec![0.0; keys.len()];
            for t in 0..tokens as usize {
                for h in 0..2 {
                    for c in 0..3 {
                        keys[t * 6 + h * 3 + c] = ((t + h + c) % 13) as f32 * 0.1;
                        values[t * 6 + h * 3 + c] = (t % 17) as f32 + h as f32 * 10.0 + c as f32;
                    }
                }
            }
            // An extreme first value exposes the off-by-one at a 1024-token window.
            values[0] = 1e5;
            let q = vec![0.0; 12];
            let reference = config
                .reference(&q, &keys, &values, tokens, position)
                .unwrap();
            let kg = stream.clone_htod(&keys).unwrap();
            let vg = stream.clone_htod(&values).unwrap();
            let mut cache =
                KvCache::new(stream.clone(), KvLayout::new(tokens, 2, 3).unwrap()).unwrap();
            cache.append(&kg, &vg).unwrap();
            let query = stream.clone_htod(&q).unwrap();
            let mut kernel = AttentionKernel::new(stream.clone(), config).unwrap();
            kernel.run(&query, &cache, position).unwrap();
            super::super::matvec::compare(&reference, &kernel.snapshot().unwrap()).unwrap();
        }
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_stable_softmax_with_nonzero_scores() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        for amplitude in [0.25, 1000.0] {
            let config = AttentionConfig::new(2, 1, 7, None, 1.0).unwrap();
            let q = vec![amplitude; 14];
            let k: Vec<f32> = (0..21).map(|i| amplitude * (i % 3) as f32).collect();
            let v: Vec<f32> = (0..21).map(|i| i as f32 * 0.1).collect();
            let expected = config.reference(&q, &k, &v, 3, 2).unwrap();
            let kg = stream.clone_htod(&k).unwrap();
            let vg = stream.clone_htod(&v).unwrap();
            let mut cache = KvCache::new(stream.clone(), KvLayout::new(3, 1, 7).unwrap()).unwrap();
            cache.append(&kg, &vg).unwrap();
            let query = stream.clone_htod(&q).unwrap();
            let mut kernel = AttentionKernel::new(stream.clone(), config).unwrap();
            kernel.run(&query, &cache, 2).unwrap();
            super::super::matvec::compare(&expected, &kernel.snapshot().unwrap()).unwrap();
        }
    }
}

#[cfg(test)]
mod reference_bits {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn attention_matches_original_kernel_bits() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let source = format!(
            "{}\n{}",
            include_str!("../../../kernels/ada/attention.cu"),
            include_str!("../tests_data/attention_reference.cu")
        );
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions { arch: Some("compute_89"), ..Default::default() },
        )
        .unwrap();
        let m = ctx.load_module(ptx).unwrap();
        let (new, old) = (m.load_function("causal_attention").unwrap(), m.load_function("ref_causal_attention").unwrap());
        let mut state = 0x6a09_e667u32;
        let mut next = |amp: f32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state % 65536) as f32 / 32768.0 - 1.0) * amp
        };
        let (qheads, kvheads) = (16u32, 8u32);
        for width in [256u32, 512, 640] {
            for tokens in [1u32, 7, 8, 9, 16, 17, 44, 63, 64, 130] {
                for window in [0u32, 16] {
                    let rnd = |n: u32, amp: f32, next: &mut dyn FnMut(f32) -> f32| (0..n).map(|_| next(amp)).collect::<Vec<_>>();
                    let q = stream.clone_htod(&rnd(qheads * width, 4.0, &mut next)).unwrap();
                    let k = stream.clone_htod(&rnd(tokens * kvheads * width, 4.0, &mut next)).unwrap();
                    let v = stream.clone_htod(&rnd(tokens * kvheads * width, 2.0, &mut next)).unwrap();
                    let position = tokens - 1;
                    let scale = 1.0f32 / (width as f32).sqrt();
                    let run = |f: &cudarc::driver::CudaFunction| {
                        let mut out = stream.alloc_zeros::<f32>((qheads * width) as usize).unwrap();
                        let mut l = stream.launch_builder(f);
                        l.arg(&q).arg(&k).arg(&v).arg(&mut out).arg(&width).arg(&qheads).arg(&kvheads)
                            .arg(&tokens).arg(&position).arg(&window).arg(&scale);
                        // SAFETY: q/k/v sized for qheads/kvheads/tokens, shared scores hold `tokens` floats.
                        unsafe {
                            l.launch(LaunchConfig { grid_dim: (qheads, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: tokens * 4 })
                        }
                        .unwrap();
                        stream.clone_dtoh(&out).unwrap()
                    };
                    let (a, b) = (run(&new), run(&old));
                    assert!(
                        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "width {width} tokens {tokens} window {window}"
                    );
                }
            }
        }
    }
}
