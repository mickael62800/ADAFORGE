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
    split: Option<SplitAttention>,
}
/// Three-kernel attention for long contexts (see attention.cu), bit-identical.
struct SplitAttention {
    /// Heads sharing each key load (1, 2 or 4).
    heads_per_warp: u32,
    scores_fn: CudaFunction,
    softmax_fn: CudaFunction,
    values_fn: CudaFunction,
    scores: CudaSlice<f32>,
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
        let split = if !parallel_exp
            && config.width() <= 512
            && std::env::var("ADAFORGE_SPLIT_ATTENTION").as_deref() != Ok("0")
        {
            let load = |name: &str| {
                module
                    .load_function(name)
                    .map_err(|e| failure("Attention scindée", e))
            };
            let heads_per_warp = (config.q_heads() / config.kv_heads()).min(4);
            Some(SplitAttention {
                heads_per_warp,
                scores_fn: load(match heads_per_warp {
                    4 => "attention_split_scores_h4",
                    2 => "attention_split_scores_h2",
                    _ => "attention_split_scores",
                })?,
                softmax_fn: load("attention_split_softmax")?,
                values_fn: load("attention_split_values")?,
                scores: stream
                    .alloc_zeros::<f32>(
                        config.q_heads() as usize * super::stack::MAX_CONTEXT as usize,
                    )
                    .map_err(|e| failure("Scores attention", e))?,
            })
        } else {
            None
        };
        Ok(Self {
            split,
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
        if tokens > super::stack::MAX_CONTEXT
            || layout.heads() != c.kv_heads()
            || layout.width() != c.width()
            || query.len() != c.q_heads() as usize * c.width() as usize
            || query.context() != self.stream.context()
            || keys.context() != self.stream.context()
        {
            return Err(DiagnosticError(
                format!("Dimensions/contexte attention incompatibles ou limite de {} tokens dépassée", super::stack::MAX_CONTEXT),
            ));
        }
        let range = c.visible_range(tokens, position).map_err(DiagnosticError)?;
        if self.split.is_some() {
            let blocks = (range.len() as u32).div_ceil(16).clamp(1, 32);
            return self.launch_split(query, keys, values, tokens, position, None, blocks);
        }
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
    /// Launches the split kernels. `state` (graph) overrides tokens/position.
    #[allow(clippy::too_many_arguments)]
    fn launch_split(
        &mut self,
        query: &CudaSlice<f32>,
        keys: &CudaSlice<f32>,
        values: &CudaSlice<f32>,
        tokens: u32,
        position: u32,
        state: Option<&CudaSlice<u32>>,
        score_blocks: u32,
    ) -> Result<(), DiagnosticError> {
        let c = self.config;
        let split = self.split.as_mut().expect("split attention");
        let (width, qheads, kvheads) = (c.width(), c.q_heads(), c.kv_heads());
        let window = c.window().unwrap_or(0);
        let scale = c.scale();
        let stride = super::stack::MAX_CONTEXT;
        let null = 0u64;
        let cfg = |x: u32, y: u32, threads: u32| LaunchConfig {
            grid_dim: (x, y, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = self.stream.launch_builder(&split.scores_fn);
        l.arg(query).arg(keys).arg(&mut split.scores).arg(&width).arg(&qheads).arg(&kvheads)
            .arg(&tokens).arg(&position).arg(&window).arg(&scale).arg(&stride);
        match state {
            Some(st) => l.arg(st),
            None => l.arg(&null),
        };
        // SAFETY: validated geometry (width <= 512), visible tokens <= MAX_CONTEXT
        // scores per head; blocks past the visible range return immediately.
        unsafe { l.launch(cfg(qheads / split.heads_per_warp, score_blocks, 256)) }
            .map_err(|e| failure("Scores attention", e))?;
        let mut l = self.stream.launch_builder(&split.softmax_fn);
        l.arg(&mut split.scores).arg(&tokens).arg(&position).arg(&window).arg(&stride);
        match state {
            Some(st) => l.arg(st),
            None => l.arg(&null),
        };
        // SAFETY: one 256-thread block per head over its own score row.
        unsafe { l.launch(cfg(qheads, 1, 256)) }.map_err(|e| failure("Softmax attention", e))?;
        let mut l = self.stream.launch_builder(&split.values_fn);
        l.arg(&split.scores).arg(values).arg(&mut self.output).arg(&width).arg(&qheads)
            .arg(&kvheads).arg(&tokens).arg(&position).arg(&window).arg(&stride);
        match state {
            Some(st) => l.arg(st),
            None => l.arg(&null),
        };
        // SAFETY: one thread per (head, channel < width); output qheads*width.
        unsafe { l.launch(cfg(qheads, width.div_ceil(64), 64)) }
            .map_err(|e| failure("Valeurs attention", e))?;
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
        if self.split.is_some() {
            let blocks = capacity.div_ceil(16).clamp(1, 32);
            return self.launch_split(query, data.keys, data.values, 0, 0, Some(state), blocks);
        }
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
        let (sc, sm, sv) = (
            m.load_function("attention_split_scores").unwrap(),
            m.load_function("attention_split_softmax").unwrap(),
            m.load_function("attention_split_values").unwrap(),
        );
        let mut state = 0x6a09_e667u32;
        let mut next = |amp: f32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state % 65536) as f32 / 32768.0 - 1.0) * amp
        };
        let sc2 = m.load_function("attention_split_scores_h2").unwrap();
        let sc4 = m.load_function("attention_split_scores_h4").unwrap();
        let qheads = 16u32;
        for (kvheads, width) in [(8u32, 256u32), (8, 512), (8, 640), (1, 512), (1, 256)] {
            for tokens in [1u32, 7, 8, 9, 16, 17, 44, 63, 64, 130, 1100, 3001] {
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
                    if width <= 512 {
                        let stride = tokens.next_multiple_of(4);
                        let mut scores = stream.alloc_zeros::<f32>((qheads * stride) as usize).unwrap();
                        let mut out = stream.alloc_zeros::<f32>((qheads * width) as usize).unwrap();
                        let null = 0u64;
                        let cfg = |x: u32, y: u32, t: u32| LaunchConfig { grid_dim: (x, y, 1), block_dim: (t, 1, 1), shared_mem_bytes: 0 };
                        let hs = (qheads / kvheads).min(4);
                        let scf = match hs { 4 => &sc4, 2 => &sc2, _ => &sc };
                        let mut l = stream.launch_builder(scf);
                        l.arg(&q).arg(&k).arg(&mut scores).arg(&width).arg(&qheads).arg(&kvheads)
                            .arg(&tokens).arg(&position).arg(&window).arg(&scale).arg(&stride).arg(&null);
                        // SAFETY: scores hold qheads*stride floats; visible tokens <= stride.
                        unsafe { l.launch(cfg(qheads / hs, 7, 256)) }.unwrap();
                        let mut l = stream.launch_builder(&sm);
                        l.arg(&mut scores).arg(&tokens).arg(&position).arg(&window).arg(&stride).arg(&null);
                        // SAFETY: one block per head row.
                        unsafe { l.launch(cfg(qheads, 1, 256)) }.unwrap();
                        let mut l = stream.launch_builder(&sv);
                        l.arg(&scores).arg(&v).arg(&mut out).arg(&width).arg(&qheads).arg(&kvheads)
                            .arg(&tokens).arg(&position).arg(&window).arg(&stride).arg(&null);
                        // SAFETY: one thread per (head, channel).
                        unsafe { l.launch(cfg(qheads, width.div_ceil(64), 64)) }.unwrap();
                        let c = stream.clone_dtoh(&out).unwrap();
                        assert!(
                            c.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "split: width {width} tokens {tokens} window {window}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod split_timing {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg, sys};
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn split_attention_kernel_times() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            include_str!("../../../kernels/ada/attention.cu"),
            cudarc::nvrtc::CompileOptions { arch: Some("compute_89"), ..Default::default() },
        )
        .unwrap();
        let m = ctx.load_module(ptx).unwrap();
        let names = ["attention_split_scores", "attention_split_softmax", "attention_split_values", "attention_split_values_h2", "attention_split_values_h4", "attention_split_scores_h2", "attention_split_scores_h4", "attention_split_scores_h8"];
        let f: Vec<_> = names.iter().map(|n| m.load_function(n).unwrap()).collect();
        for (tokens, kvheads, width, window) in [(7620u32, 8u32, 256u32, 1024u32), (7620, 1, 512, 0)] {
            let qheads = 16u32;
            let stride = 10240u32;
            let q = stream.clone_htod(&vec![0.01f32; (qheads * width) as usize]).unwrap();
            let kv: Vec<f32> = (0..tokens * kvheads * width).map(|i| (i % 97) as f32 * 0.001).collect();
            let k = stream.clone_htod(&kv).unwrap();
            let v = stream.clone_htod(&kv).unwrap();
            let mut scores = stream.alloc_zeros::<f32>((qheads * stride) as usize).unwrap();
            let mut out = stream.alloc_zeros::<f32>((qheads * width) as usize).unwrap();
            let (position, scale, null) = (tokens - 1, 0.0625f32, 0u64);
            let cfg = |x: u32, y: u32, t: u32| LaunchConfig { grid_dim: (x, y, 1), block_dim: (t, 1, 1), shared_mem_bytes: 0 };
            for (fi, hs) in [(5usize, 2u32), (6, 4), (7, 8)] {
                if hs > qheads / kvheads { continue; }
                let mut best = f32::MAX;
                for _ in 0..10 {
                    let e0 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    let mut l = stream.launch_builder(&f[fi]);
                    l.arg(&q).arg(&k).arg(&mut scores).arg(&width).arg(&qheads).arg(&kvheads)
                        .arg(&tokens).arg(&position).arg(&window).arg(&scale).arg(&stride).arg(&null);
                    // SAFETY: buffers sized above.
                    unsafe { l.launch(cfg(qheads / hs, 32, 256)) }.unwrap();
                    let e1 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    best = best.min(e0.elapsed_ms(&e1).unwrap());
                }
                println!("SCORES h{hs} kv{kvheads}: {:.1}us", best * 1e3);
            }
            let mut times = [f32::MAX; 3];
            for _ in 0..20 {
                let e = |s: &std::sync::Arc<cudarc::driver::CudaStream>| s.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                let e0 = e(&stream);
                let mut l = stream.launch_builder(&f[0]);
                l.arg(&q).arg(&k).arg(&mut scores).arg(&width).arg(&qheads).arg(&kvheads)
                    .arg(&tokens).arg(&position).arg(&window).arg(&scale).arg(&stride).arg(&null);
                // SAFETY: buffers sized above.
                unsafe { l.launch(cfg(qheads, 32, 256)) }.unwrap();
                let e1 = e(&stream);
                let mut l = stream.launch_builder(&f[1]);
                l.arg(&mut scores).arg(&tokens).arg(&position).arg(&window).arg(&stride).arg(&null);
                // SAFETY: one block per head row.
                unsafe { l.launch(cfg(qheads, 1, 256)) }.unwrap();
                let e2 = e(&stream);
                let hpt = 1u32;
                let vf = &f[match hpt { 1 => 2, 2 => 3, _ => 4 }];
                let mut l = stream.launch_builder(vf);
                l.arg(&scores).arg(&v).arg(&mut out).arg(&width).arg(&qheads).arg(&kvheads)
                    .arg(&tokens).arg(&position).arg(&window).arg(&stride).arg(&null);
                // SAFETY: one thread per (head group, channel).
                unsafe { l.launch(cfg(qheads / hpt, width.div_ceil(64), 64)) }.unwrap();
                let e3 = e(&stream);
                times[0] = times[0].min(e0.elapsed_ms(&e1).unwrap());
                times[1] = times[1].min(e1.elapsed_ms(&e2).unwrap());
                times[2] = times[2].min(e2.elapsed_ms(&e3).unwrap());
            }
            println!("SPLIT kv{kvheads} w{width} window{window}: scores {:.1}us softmax {:.1}us values {:.1}us",
                times[0] * 1e3, times[1] * 1e3, times[2] * 1e3);
        }
    }
}
