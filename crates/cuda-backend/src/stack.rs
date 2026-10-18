//! Bloc CUDA résident réutilisable et diagnostic de blocs consécutifs.
use super::{
    attention::AttentionKernel, attention_output::OutputKernel, embedding::GpuEmbeddings, failure,
    ffn::FfnKernel, kv_cache::KvCache,
};
use adaforge_application::{DiagnosticError, LayerWeights, OutputBranchReport, StackReport};
use adaforge_domain::{EmbeddingBatch, KvLayout, RmsNorm};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::sync::Arc;

// Diagnostic-only profiler windows: skip the benchmark warmup by default.
static PROFILE_GENERATIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct ResidentLayer {
    stream: Arc<CudaStream>,
    rms: CudaFunction,
    project_qkv: CudaFunction,
    qkv_kparallel: bool,
    shared_kv: u32,
    prep_cache: bool,
    prepare_qkv_cache_dynamic: CudaFunction,
    grouped_qkv_prep: bool,
    prepare_qkv: CudaFunction,
    prepare_qkv_dynamic: CudaFunction,
    fused_norm_rope: bool,
    norm_rotary: CudaFunction,
    norm_rotary_dynamic: CudaFunction,
    rotary: CudaFunction,
    rotary_dynamic: CudaFunction,
    cache_dynamic: CudaFunction,
    matrices: Vec<CudaSlice<u8>>,
    scales: Vec<CudaSlice<f32>>,
    freq: CudaSlice<f64>,
    normalized: CudaSlice<f32>,
    raw: Vec<CudaSlice<f32>>,
    heads: Vec<CudaSlice<f32>>,
    rotated: Vec<CudaSlice<f32>>,
    width: u32,
    head_width: u32,
    rows: [u32; 3],
    eps: [f32; 4],
    context: u32,
    next_position: u32,
    capacity: u32,
    poisoned: bool,
    cache: KvCache,
    attention: AttentionKernel,
    output_branch: OutputKernel,
    ffn: FfnKernel,
}
fn validate(w: &LayerWeights<'_>, width: u32, tokens: u32) -> Result<(), DiagnosticError> {
    let hw = w.rope.width();
    if tokens == 0
        || tokens > 512
        || tokens > w.rope.context()
        || w.input_norm.weights().len() != width as usize
        || w.matrices.iter().any(|m| m.columns() != width)
        || w.head_norms
            .iter()
            .any(|n| n.weights().len() != hw as usize)
        || w.matrices.iter().any(|m| !m.rows().is_multiple_of(hw))
        || w.matrices[0].rows() != w.attention.q_heads() * hw
        || w.matrices[1].rows() != w.attention.kv_heads() * hw
        || w.matrices[2].rows() != w.matrices[1].rows()
        || w.attention.width() != hw
        || w.output.matrix().columns() != w.matrices[0].rows()
        || w.output.matrix().rows() != width
        || w.ffn.width() != width
    {
        return Err(DiagnosticError(
            "Dimensions de blocs consécutifs incompatibles ou contexte insuffisant".into(),
        ));
    }
    Ok(())
}
impl ResidentLayer {
    fn new(
        stream: Arc<CudaStream>,
        w: &LayerWeights<'_>,
        tokens: u32,
    ) -> Result<Self, DiagnosticError> {
        let width = w.matrices[0].columns();
        validate(w, width, tokens)?;
        let source = format!(
            "{}\n{}\n{}",
            include_str!("../../../kernels/ada/rms_norm.cu"),
            include_str!("../../../kernels/ada/q4_matvec.cu"),
            include_str!("../../../kernels/ada/rope.cu")
        );
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| failure("Compilation bloc résident", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| failure("Module bloc", e))?;
        let load = |name| {
            module
                .load_function(name)
                .map_err(|e| failure("Kernel bloc", e))
        };
        let rms = load("rms_norm")?;
        let qkv_kparallel = matches!(
            std::env::var("ADAFORGE_EXPERIMENTAL_KPAR_PROJECTIONS").as_deref(),
            Ok("1" | "qkv")
        );
        let project_qkv = load(if qkv_kparallel {
            "q4_qkv_kparallel"
        } else {
            "q4_qkv"
        })?;
        let rotary = load("rope_neox")?;
        let rotary_dynamic = load("rope_dynamic")?;
        let cache_dynamic = load("cache_dynamic")?;
        let matrices = w
            .matrices
            .iter()
            .map(|m| {
                stream
                    .clone_htod(m.bytes())
                    .map_err(|e| failure("Poids bloc", e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ones = vec![1.0; w.rope.width() as usize];
        let scales = [
            w.input_norm.weights(),
            w.head_norms[0].weights(),
            w.head_norms[1].weights(),
            &ones,
        ]
        .iter()
        .map(|s| stream.clone_htod(*s).map_err(|e| failure("Normes bloc", e)))
        .collect::<Result<Vec<_>, _>>()?;
        let alloc = |n: u32| {
            stream
                .alloc_zeros::<f32>(n as usize)
                .map_err(|e| failure("Buffer bloc", e))
        };
        let rows = w.matrices.map(|m| m.rows());
        let raw = rows
            .iter()
            .map(|&n| alloc(n))
            .collect::<Result<Vec<_>, _>>()?;
        let heads = rows
            .iter()
            .map(|&n| alloc(n))
            .collect::<Result<Vec<_>, _>>()?;
        let rotated = rows[..2]
            .iter()
            .map(|&n| alloc(n))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rms,
            project_qkv,
            qkv_kparallel,
            shared_kv: u32::from(std::ptr::eq(w.matrices[1], w.matrices[2])),
            prep_cache: std::env::var("ADAFORGE_EXPERIMENTAL_PREP_CACHE").as_deref() == Ok("1"),
            prepare_qkv_cache_dynamic: load("prepare_qkv_cache_dynamic")?,
            grouped_qkv_prep: std::env::var("ADAFORGE_EXPERIMENTAL_QKV_PREP").as_deref() != Ok("0")
                && matches!(w.rope.width(), 256 | 512),
            prepare_qkv: load("prepare_qkv")?,
            prepare_qkv_dynamic: load("prepare_qkv_dynamic")?,
            fused_norm_rope: std::env::var("ADAFORGE_EXPERIMENTAL_NORM_ROPE").as_deref() == Ok("1")
                && matches!(w.rope.width(), 256 | 512),
            norm_rotary: load("norm_rope")?,
            norm_rotary_dynamic: load("norm_rope_dynamic")?,
            rotary,
            rotary_dynamic,
            cache_dynamic,
            matrices,
            scales,
            freq: stream
                .clone_htod(w.rope.inverse_frequencies())
                .map_err(|e| failure("Fréquences bloc", e))?,
            normalized: alloc(width)?,
            raw,
            heads,
            rotated,
            width,
            head_width: w.rope.width(),
            rows,
            eps: [
                w.input_norm.epsilon(),
                w.head_norms[0].epsilon(),
                w.head_norms[1].epsilon(),
                w.input_norm.epsilon(),
            ],
            context: w.rope.context(),
            next_position: 0,
            capacity: tokens,
            poisoned: false,
            cache: KvCache::new(
                stream.clone(),
                KvLayout::new(tokens, w.attention.kv_heads(), w.rope.width())
                    .map_err(DiagnosticError)?,
            )?,
            attention: AttentionKernel::new(stream.clone(), w.attention)?,
            output_branch: OutputKernel::new(stream.clone(), w.output)?,
            ffn: FfnKernel::new(stream.clone(), w.ffn)?,
            stream,
        })
    }
    fn run(&mut self, input: &CudaSlice<f32>, position: u32) -> Result<(), DiagnosticError> {
        if self.poisoned
            || input.len() != self.width as usize
            || input.context() != self.stream.context()
            || position != self.next_position
            || position >= self.context
            || position >= self.capacity
        {
            return Err(DiagnosticError(
                "Entrée, position ou état du bloc résident invalide".into(),
            ));
        }
        // After a CUDA failure a partial cache update may have occurred. Never reuse it.
        self.poisoned = true;
        self.launch_body(input, position, None)?;
        self.next_position += 1;
        self.poisoned = false;
        Ok(())
    }
    fn launch_body(
        &mut self,
        input: &CudaSlice<f32>,
        position: u32,
        state: Option<&CudaSlice<u32>>,
    ) -> Result<(), DiagnosticError> {
        norm(
            &self.stream,
            &self.rms,
            input,
            &self.scales[0],
            &mut self.normalized,
            self.width,
            self.eps[0],
        )?;
        let total =
            self.rows[0] + self.rows[1] + if self.shared_kv != 0 { 0 } else { self.rows[2] };
        let mut launch = self.stream.launch_builder(&self.project_qkv);
        let (q, kv) = self.raw.split_at_mut(1);
        let (k, v) = kv.split_at_mut(1);
        launch
            .arg(&self.matrices[0])
            .arg(&self.matrices[1])
            .arg(&self.matrices[2])
            .arg(&self.normalized)
            .arg(&mut q[0])
            .arg(&mut k[0])
            .arg(&mut v[0])
            .arg(&self.width)
            .arg(&self.rows[0])
            .arg(&self.rows[1])
            .arg(&self.shared_kv);
        // SAFETY: validated Q/K/V geometries and distinct outputs; the grid
        // matches the selected one-warp or four-warps-per-row function.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    if self.qkv_kparallel {
                        total
                    } else {
                        total.div_ceil(4)
                    },
                    1,
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Projection QKV fusionnée", e))?;
        if self.grouped_qkv_prep {
            let qheads = self.rows[0] / self.head_width;
            let kvheads = self.rows[1] / self.head_width;
            let vsource = if self.shared_kv != 0 { 1 } else { 2 };
            let (qout, kout) = self.rotated.split_at_mut(1);
            let into_cache = self.prep_cache && state.is_some();
            let (keyout, valueout) = if into_cache {
                self.cache.graph_buffers()?
            } else {
                (&mut kout[0], &mut self.heads[2])
            };
            let mut launch = self.stream.launch_builder(if into_cache {
                &self.prepare_qkv_cache_dynamic
            } else if state.is_some() {
                &self.prepare_qkv_dynamic
            } else {
                &self.prepare_qkv
            });
            launch
                .arg(&self.raw[0])
                .arg(&self.raw[1])
                .arg(&self.raw[vsource])
                .arg(&self.scales[1])
                .arg(&self.scales[2])
                .arg(&self.scales[3])
                .arg(&self.freq)
                .arg(&mut qout[0])
                .arg(keyout)
                .arg(valueout)
                .arg(&self.head_width)
                .arg(&qheads)
                .arg(&kvheads)
                .arg(&self.eps[1])
                .arg(&self.eps[2])
                .arg(&self.eps[3]);
            if let Some(state) = state {
                launch.arg(state);
            } else {
                launch.arg(&position);
            }
            // SAFETY: validated heads of width 256/512, one 256-thread block per Q/K/V head, disjoint outputs; shared K/V uses a read-only alias.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (qheads + 2 * kvheads, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| failure("Préparation QKV groupée", e))?;
        } else {
            for i in 0..3 {
                if self.fused_norm_rope && i < 2 {
                    continue;
                }
                let source = if i == 2 && self.shared_kv != 0 { 1 } else { i };
                norm(
                    &self.stream,
                    &self.rms,
                    &self.raw[source],
                    &self.scales[i + 1],
                    &mut self.heads[i],
                    self.head_width,
                    self.eps[i + 1],
                )?;
            }
            for i in 0..2 {
                if self.fused_norm_rope {
                    let mut launch = self.stream.launch_builder(if state.is_some() {
                        &self.norm_rotary_dynamic
                    } else {
                        &self.norm_rotary
                    });
                    launch
                        .arg(&self.raw[i])
                        .arg(&self.scales[i + 1])
                        .arg(&self.freq)
                        .arg(&mut self.rotated[i])
                        .arg(&self.head_width)
                        .arg(&self.eps[i + 1]);
                    if let Some(state) = state {
                        launch.arg(state);
                    } else {
                        launch.arg(&position);
                    }
                    // SAFETY: validated width 256/512, 256 threads per head, separate full-size outputs; graph state is session-owned.
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (self.rows[i] / self.head_width, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map_err(|e| failure("Norme/RoPE fusionnée", e))?;
                    continue;
                }
                let elements = self.rows[i];
                let config = LaunchConfig {
                    grid_dim: ((elements / 2).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut launch = self.stream.launch_builder(if state.is_some() {
                    &self.rotary_dynamic
                } else {
                    &self.rotary
                });
                launch
                    .arg(&self.heads[i])
                    .arg(&self.freq)
                    .arg(&mut self.rotated[i])
                    .arg(&self.head_width)
                    .arg(&elements);
                if let Some(state) = state {
                    launch.arg(state);
                } else {
                    launch.arg(&position);
                }
                // SAFETY: validated even head width; independent input/output, one thread per pair.
                unsafe { launch.launch(config) }.map_err(|e| failure("RoPE bloc résident", e))?;
            }
        }
        if let Some(state) = state {
            if !(self.prep_cache && self.grouped_qkv_prep) {
                let (keys, values) = self.cache.graph_buffers()?;
                let elements = self.rows[1];
                let mut launch = self.stream.launch_builder(&self.cache_dynamic);
                launch
                    .arg(&self.rotated[1])
                    .arg(&self.heads[2])
                    .arg(keys)
                    .arg(values)
                    .arg(&elements)
                    .arg(state);
                // SAFETY: graph state position checked against capacity before launch; distinct cache buffers.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (elements.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| failure("Écriture KV graphe", e))?;
            }
            self.attention
                .run_graph(&self.rotated[0], &self.cache, state, self.capacity)?;
        } else {
            self.cache.append(&self.rotated[1], &self.heads[2])?;
            self.attention
                .run(&self.rotated[0], &self.cache, position)?;
        }
        self.output_branch.run(self.attention.output(), input)?;
        self.ffn.run(self.output_branch.output())?;
        Ok(())
    }
    fn output(&self) -> &CudaSlice<f32> {
        self.ffn.output()
    }
}
fn norm(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    input: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    width: u32,
    eps: f32,
) -> Result<(), DiagnosticError> {
    let config = LaunchConfig {
        grid_dim: ((input.len() / width as usize) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(function);
    launch
        .arg(input)
        .arg(weights)
        .arg(output)
        .arg(&width)
        .arg(&eps);
    // SAFETY: internally allocated, validated sizes, one block per vector/head, distinct output.
    unsafe { launch.launch(config) }.map_err(|e| failure("Norme bloc résident", e))?;
    Ok(())
}
fn fp32(values: Vec<f64>) -> Vec<f32> {
    values.into_iter().map(|x| x as f32).collect()
}
#[derive(Default)]
struct ReferenceCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}
fn reference(
    w: &LayerWeights<'_>,
    x: &[f32],
    cache: &mut ReferenceCache,
    position: u32,
) -> Result<Vec<f64>, DiagnosticError> {
    let compute = || -> Result<Vec<Vec<f32>>, String> {
        let normalized = fp32(w.input_norm.reference(x)?);
        let mut qkv = w
            .matrices
            .iter()
            .map(|m| m.reference_matvec(&normalized).map(fp32))
            .collect::<Result<Vec<_>, _>>()?;
        for (i, item) in qkv.iter_mut().enumerate().take(2) {
            *item = fp32(w.head_norms[i].reference(item)?);
            *item = fp32(w.rope.reference(item, position)?);
        }
        let vn = RmsNorm::new(vec![1.0; w.rope.width() as usize], w.input_norm.epsilon())?;
        qkv[2] = fp32(vn.reference(&qkv[2])?);
        Ok(qkv)
    };
    let qkv = compute().map_err(DiagnosticError)?;
    cache.keys.extend_from_slice(&qkv[1]);
    cache.values.extend_from_slice(&qkv[2]);
    let a = w
        .attention
        .reference(&qkv[0], &cache.keys, &cache.values, position + 1, position)
        .map_err(DiagnosticError)?;
    let branch = w.output.reference(&fp32(a), x).map_err(DiagnosticError)?;
    w.ffn.reference(&fp32(branch)).map_err(DiagnosticError)
}
pub(crate) fn check(
    weights: &[LayerWeights<'_>],
    embeddings: &EmbeddingBatch,
) -> Result<StackReport, DiagnosticError> {
    check_with_head(weights, embeddings, None)
}
pub(crate) fn check_with_head(
    weights: &[LayerWeights<'_>],
    embeddings: &EmbeddingBatch,
    head: Option<adaforge_application::OutputHead<'_>>,
) -> Result<StackReport, DiagnosticError> {
    let tokens = embeddings.token_ids().len() as u32;
    if tokens > 32 || weights.is_empty() || weights.len() > 48 {
        return Err(DiagnosticError(
            "Ce diagnostic exige entre un et 48 blocs consécutifs".into(),
        ));
    }
    for w in weights {
        validate(w, embeddings.width(), tokens)?;
    }
    if let Some(h) = head.as_ref()
        && (weights.len() != 48
            || h.matrix.columns() != embeddings.width()
            || h.norm.weights().len() != embeddings.width() as usize
            || !h.cap.is_finite()
            || h.cap <= 0.0)
    {
        return Err(DiagnosticError("Tête de sortie incompatible".into()));
    }
    let ctx = CudaContext::new(0).map_err(|e| failure("CUDA blocs", e))?;
    let name = ctx.name().map_err(|e| failure("Nom GPU", e))?;
    let major = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| failure("CUDA", e))?;
    let minor = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| failure("CUDA", e))?;
    if !name.contains("RTX 4090") || (major, minor) != (8, 9) {
        return Err(DiagnosticError("RTX 4090 sm_89 requise".into()));
    }
    let stream = ctx.default_stream();
    let gpu_embeddings = GpuEmbeddings::new(stream.clone(), embeddings)?;
    let mut input = stream
        .alloc_zeros::<f32>(embeddings.width() as usize)
        .map_err(|e| failure("Entrée blocs", e))?;
    let mut layers = weights
        .iter()
        .map(|w| ResidentLayer::new(stream.clone(), w, tokens))
        .collect::<Result<Vec<_>, _>>()?;
    let mut caches = weights
        .iter()
        .map(|_| ReferenceCache::default())
        .collect::<Vec<_>>();
    let mut reports = weights
        .iter()
        .map(|_| OutputBranchReport {
            checked_values: 0,
            width: embeddings.width(),
            max_absolute_error: 0.0,
            max_tolerance_ratio: 0.0,
        })
        .collect::<Vec<_>>();
    let cpu_embeddings = embeddings.reference(true);
    let mut last_cpu = Vec::new();
    for position in 0..tokens {
        gpu_embeddings.copy_row_to(position as usize, &mut input)?;
        layers[0].run(&input, position)?;
        for index in 1..layers.len() {
            let (previous, next) = layers.split_at_mut(index);
            // Borrow the prior resident output directly; no host activation transfer.
            next[0].run(previous[index - 1].output(), position)?;
        }
        let width = embeddings.width() as usize;
        let mut x =
            cpu_embeddings[position as usize * width..(position as usize + 1) * width].to_vec();
        for index in 0..layers.len() {
            let expected = reference(&weights[index], &x, &mut caches[index], position)?;
            let actual = layers[index].ffn.snapshot()?;
            let (error, ratio) = super::matvec::compare(&expected, &actual)
                .map_err(|e| DiagnosticError(format!("Bloc {index}, position {position} : {e}")))?;
            let stats = &mut reports[index];
            stats.checked_values += width;
            stats.max_absolute_error = stats.max_absolute_error.max(error);
            stats.max_tolerance_ratio = stats.max_tolerance_ratio.max(ratio);
            x = fp32(expected);
        }
        last_cpu = x;
    }
    let logits = head
        .map(|h| {
            super::logits::check(
                stream.clone(),
                h,
                layers.last().expect("nonempty").output(),
                &last_cpu,
                tokens - 1,
            )
        })
        .transpose()?;
    Ok(StackReport {
        logits,
        token_ids: embeddings.token_ids().to_vec(),
        width: embeddings.width(),
        layers: reports,
        geometry: weights
            .iter()
            .map(|w| adaforge_application::LayerGeometry {
                q_heads: w.attention.q_heads(),
                kv_heads: w.attention.kv_heads(),
                head_width: w.rope.width(),
                window: w.attention.window(),
                shared_kv_weights: std::ptr::eq(w.matrices[1], w.matrices[2]),
                attenuated_pairs: w.rope.factors().iter().filter(|&&f| f > 1e20).count(),
            })
            .collect(),
    })
}

pub(crate) fn generate(
    weights: &[LayerWeights<'_>],
    head: adaforge_application::OutputHead<'_>,
    prompt: &[u32],
    stop: &[u32],
    max_new: u32,
) -> Result<adaforge_application::GenerationTokens, DiagnosticError> {
    if prompt.len() > 256 || !(1..=128).contains(&max_new) {
        return Err(DiagnosticError(
            "Configuration de génération invalide".into(),
        ));
    }
    let mut session = open_session(weights, head, prompt.len() as u32 + max_new)?;
    session.generate(prompt, stop, max_new)
}
pub(crate) fn open_session(
    weights: &[LayerWeights<'_>],
    head: adaforge_application::OutputHead<'_>,
    capacity: u32,
) -> Result<Box<dyn adaforge_application::GenerationSession>, DiagnosticError> {
    if weights.len() != 48 || capacity == 0 || capacity > 512 {
        return Err(DiagnosticError(
            "Session limitée à 48 blocs et 512 tokens de contexte".into(),
        ));
    }
    let vocabulary = head.matrix.rows();
    let width = head.matrix.columns();
    if head.norm.weights().len() != width as usize {
        return Err(DiagnosticError("Norme de sortie incompatible".into()));
    }
    for w in weights {
        validate(w, width, capacity)?;
    }
    let ctx = CudaContext::new(0).map_err(|e| failure("CUDA génération", e))?;
    let name = ctx.name().map_err(|e| failure("Nom GPU", e))?;
    let major = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| failure("CUDA", e))?;
    let minor = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| failure("CUDA", e))?;
    if !name.contains("RTX 4090") || (major, minor) != (8, 9) {
        return Err(DiagnosticError("RTX 4090 sm_89 requise".into()));
    }
    // SAFETY: this session owns all its resources and uses exactly one private stream.
    // Explicit stream ordering covers all kernel, copy, graph and host read dependencies.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream().map_err(|e| failure("Stream session", e))?;
    let layers = weights
        .iter()
        .map(|w| ResidentLayer::new(stream.clone(), w, capacity))
        .collect::<Result<Vec<_>, _>>()?;
    let head = super::logits::ResidentHead::new(stream.clone(), head)?;
    let input = stream
        .alloc_zeros::<f32>(width as usize)
        .map_err(|e| failure("Entrée génération", e))?;
    let state = stream
        .alloc_zeros::<u32>(2)
        .map_err(|e| failure("État token/position", e))?;
    let mut session = CudaGenerationSession {
        graph: None,
        state,
        layers,
        head,
        input,
        cached: Vec::new(),
        capacity,
        vocabulary,
        poisoned: false,
    };
    if std::env::var_os("ADAFORGE_DISABLE_GRAPHS").is_none() {
        session.capture_graph()?;
    }
    if std::env::var("ADAFORGE_DIAGNOSTICS").as_deref() == Ok("1") {
        let qkv = session.layers.iter().filter(|l| l.qkv_kparallel).count();
        let output = session
            .layers
            .iter()
            .filter(|l| l.output_branch.kparallel_enabled())
            .count();
        let tiled = session
            .layers
            .iter()
            .filter(|l| l.ffn.strategy_name() == "q4_tiled")
            .count();
        let q8 = session
            .layers
            .iter()
            .filter(|l| l.ffn.strategy_name() == "q8")
            .count();
        let bytes: usize = session
            .layers
            .iter()
            .map(|l| l.ffn.matrix_weight_bytes())
            .sum();
        eprintln!(
            "ADAFORGE_CONFIG {{\"backend\":\"cuda-sm89\",\"layers\":{},\"context_capacity\":{},\"vocabulary\":{},\"graphs\":{},\"kv_dtype\":\"f32\",\"tiled_ffn_layers\":{},\"q8_ffn_layers\":{},\"qkv_kparallel_layers\":{},\"output_kparallel_layers\":{},\"fused_norm_rope_layers\":{},\"grouped_qkv_prep_layers\":{},\"prep_cache_layers\":{},\"parallel_exp_layers\":{},\"parallel_exp_min_visible\":64,\"ffn_matrix_bytes\":{}}}",
            session.layers.len(),
            capacity,
            vocabulary,
            session.graph.is_some(),
            tiled,
            q8,
            qkv,
            output,
            session
                .layers
                .iter()
                .filter(|layer| layer.fused_norm_rope)
                .count(),
            session
                .layers
                .iter()
                .filter(|layer| layer.grouped_qkv_prep)
                .count(),
            session
                .layers
                .iter()
                .filter(|layer| layer.prep_cache
                    && layer.grouped_qkv_prep
                    && session.graph.is_some())
                .count(),
            session
                .layers
                .iter()
                .filter(|layer| layer.attention.parallel_exp_enabled())
                .count(),
            bytes
        );
    }
    Ok(Box::new(session))
}
struct CudaGenerationSession {
    // Drop the graph before its captured buffers/modules; all resources below
    // remain owned by this non-Send session and share one CUDA stream.
    graph: Option<cudarc::driver::CudaGraph>,
    state: CudaSlice<u32>,
    layers: Vec<ResidentLayer>,
    head: super::logits::ResidentHead,
    input: CudaSlice<f32>,
    cached: Vec<u32>,
    capacity: u32,
    vocabulary: u32,
    poisoned: bool,
}
impl CudaGenerationSession {
    fn capture_graph(&mut self) -> Result<(), DiagnosticError> {
        let stream = self.layers[0].stream.clone();
        stream
            .synchronize()
            .map_err(|e| failure("Préparation graphe", e))?;
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| failure("Début capture CUDA", e))?;
        let captured = (|| {
            self.head.embed_graph(&self.state, &mut self.input)?;
            self.layers[0].launch_body(&self.input, 0, Some(&self.state))?;
            for index in 1..self.layers.len() {
                let (previous, next) = self.layers.split_at_mut(index);
                next[0].launch_body(previous[index - 1].output(), 0, Some(&self.state))?;
            }
            Ok::<(), DiagnosticError>(())
        })();
        // End capture even on failure, so the stream is never left capturing.
        let graph = stream
            .end_capture(sys::CUgraphInstantiate_flags(0))
            .map_err(|e| failure("Fin capture CUDA", e))?;
        captured?;
        let graph = graph.ok_or_else(|| DiagnosticError("Graphe CUDA vide".into()))?;
        graph
            .upload()
            .map_err(|e| failure("Préchargement graphe", e))?;
        stream
            .synchronize()
            .map_err(|e| failure("Préparation exécution graphe", e))?;
        self.graph = Some(graph);
        Ok(())
    }
    fn consume(&mut self, id: u32) -> Result<(), DiagnosticError> {
        let position = self.cached.len() as u32;
        if id >= self.vocabulary || position >= self.capacity {
            return Err(DiagnosticError("Token/position hors capacité".into()));
        }
        if let Some(graph) = self.graph.as_ref() {
            self.layers[0]
                .stream
                .memcpy_htod(&[id, position], &mut self.state)
                .map_err(|e| failure("Mise à jour état graphe", e))?;
            graph
                .launch()
                .map_err(|e| failure("Exécution graphe CUDA", e))?;
            for layer in &mut self.layers {
                layer.cache.graph_commit(position)?;
                layer.next_position = position + 1;
            }
        } else {
            self.head.embed(id, &mut self.input)?;
            run_layers(&mut self.layers, &self.input, position)?;
        }
        self.cached.push(id);
        Ok(())
    }
}
impl CudaGenerationSession {
    fn generate_stream_impl(
        &mut self,
        prompt: &[u32],
        stop: &[u32],
        max_new: u32,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<adaforge_application::GenerationTokens, DiagnosticError> {
        if self.poisoned {
            return Err(DiagnosticError(
                "Session CUDA invalide ; redémarrer le chat".into(),
            ));
        }
        if prompt.is_empty()
            || !(1..=128).contains(&max_new)
            || prompt.len() + max_new as usize > self.capacity as usize
            || stop.is_empty()
            || prompt.iter().chain(stop).any(|&id| id >= self.vocabulary)
            || !prompt.starts_with(&self.cached)
        {
            return Err(DiagnosticError(
                "Prompt, préfixe KV ou capacité de session incompatible".into(),
            ));
        }
        let started = std::time::Instant::now();
        let cached_tokens = self.cached.len();
        self.poisoned = true;
        for &id in &prompt[self.cached.len()..] {
            self.consume(id)?;
        }
        self.layers[0]
            .stream
            .synchronize()
            .map_err(|e| failure("Synchronisation prefill", e))?;
        let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
        let _decode_profile = if std::env::var("ADAFORGE_PROFILE_DECODE").as_deref() == Ok("1") {
            let skip: usize = std::env::var("ADAFORGE_PROFILE_SKIP")
                .unwrap_or_else(|_| "1".into())
                .parse()
                .map_err(|_| DiagnosticError("ADAFORGE_PROFILE_SKIP invalide".into()))?;
            let request = PROFILE_GENERATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if request >= skip {
                Some(
                    cudarc::driver::Profiler::new()
                        .map_err(|e| failure("Début profil decode", e))?,
                )
            } else {
                None
            }
        } else {
            None
        };
        let mut first_ready = None;
        let mut last_ready = None;
        let mut ids = Vec::new();
        let mut stopped = false;
        // Diagnostic : temps GPU par token (événements CUDA), tête vs 48 blocs.
        let timing = std::env::var("ADAFORGE_TIMING_DECODE").as_deref() == Ok("1");
        let timing_stream = self.layers[0].stream.clone();
        let timed = |s: &Arc<CudaStream>| {
            s.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| failure("Événement timing", e))
        };
        let (mut head_ms, mut blocks_ms, mut timed_steps) = (Vec::new(), Vec::new(), 0usize);
        for _ in 0..max_new {
            let e0 = if timing { Some(timed(&timing_stream)?) } else { None };
            let id = self
                .head
                .select(self.layers.last().expect("48 layers").output())?;
            let e1 = if timing { Some(timed(&timing_stream)?) } else { None };
            if stop.contains(&id) {
                stopped = true;
                break;
            }
            let ready = started.elapsed().as_secs_f64() * 1000.0;
            if first_ready.is_none() {
                first_ready = Some(ready);
            }
            last_ready = Some(ready);
            self.consume(id)?;
            if let (Some(e0), Some(e1)) = (e0, e1) {
                let e2 = timed(&timing_stream)?;
                head_ms.push(e0.elapsed_ms(&e1).map_err(|e| failure("Timing tête", e))?);
                blocks_ms.push(e1.elapsed_ms(&e2).map_err(|e| failure("Timing blocs", e))?);
                timed_steps += 1;
            }
            ids.push(id);
            on_token(id);
        }
        if timing && timed_steps > 1 {
            // Ignore le premier pas (chauffe).
            let avg = |v: &[f32]| v[1..].iter().sum::<f32>() / (v.len() - 1) as f32;
            let (h, b) = (avg(&head_ms), avg(&blocks_ms));
            eprintln!(
                "DECODE_TIMING {{\"steps\":{},\"prefill_ms\":{:.3},\"head_q6k_greedy_ms\":{:.4},\"embed_48_blocks_ms\":{:.4},\"token_gpu_ms\":{:.4},\"gpu_bound_tok_s\":{:.2}}}",
                timed_steps - 1, prefill_ms, h, b, h + b, 1000.0 / (h + b)
            );
        }
        self.poisoned = false;
        self.layers[0]
            .stream
            .synchronize()
            .map_err(|e| failure("Synchronisation fin génération", e))?;
        let stats = adaforge_application::GenerationStats {
            prompt_tokens: prompt.len() - cached_tokens,
            cached_tokens,
            prefill_ms,
            first_token_ms: first_ready,
            decode_tokens: ids.len().saturating_sub(1),
            decode_ms: first_ready.zip(last_ready).map_or(0.0, |(a, b)| b - a),
            total_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
        Ok(adaforge_application::GenerationTokens {
            ids,
            stopped,
            stats: Some(stats),
        })
    }
}
impl adaforge_application::GenerationSession for CudaGenerationSession {
    fn generate(
        &mut self,
        prompt: &[u32],
        stop: &[u32],
        max_new: u32,
    ) -> Result<adaforge_application::GenerationTokens, DiagnosticError> {
        self.generate_stream_impl(prompt, stop, max_new, &mut |_| {})
    }
    fn generate_stream(
        &mut self,
        prompt: &[u32],
        stop: &[u32],
        max_new: u32,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<adaforge_application::GenerationTokens, DiagnosticError> {
        self.generate_stream_impl(prompt, stop, max_new, on_token)
    }
    fn reset(&mut self) -> Result<(), DiagnosticError> {
        if self.poisoned {
            return Err(DiagnosticError(
                "Session CUDA invalide ; redémarrer le chat".into(),
            ));
        }
        self.poisoned = true;
        for layer in &mut self.layers {
            layer.cache.reset()?;
            layer.next_position = 0;
        }
        self.cached.clear();
        self.poisoned = false;
        Ok(())
    }
}
fn run_layers(
    layers: &mut [ResidentLayer],
    input: &CudaSlice<f32>,
    position: u32,
) -> Result<(), DiagnosticError> {
    layers[0].run(input, position)?;
    for index in 1..layers.len() {
        let (previous, next) = layers.split_at_mut(index);
        next[0].run(previous[index - 1].output(), position)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn fused_norm_rope_matches_separate_bits_at_large_positions() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let source = format!(
            "{}\n{}",
            include_str!("../../../kernels/ada/rms_norm.cu"),
            include_str!("../../../kernels/ada/rope.cu")
        );
        let module = ctx
            .load_module(
                compile_ptx_with_opts(
                    source,
                    CompileOptions {
                        arch: Some("compute_89"),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let rms = module.load_function("rms_norm").unwrap();
        let rope = module.load_function("rope_neox").unwrap();
        for width in [256u32, 512] {
            let elements = width * 3;
            let x = stream
                .clone_htod(
                    &(0..elements)
                        .map(|i| (i % 31) as f32 / 7.0 - 2.0)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let scales = stream
                .clone_htod(
                    &(0..width)
                        .map(|i| if i % 2 == 0 { 0.75f32 } else { -1.25 })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let freq = stream
                .clone_htod(
                    &(0..width / 2)
                        .map(|i| 10000f64.powf(-2.0 * f64::from(i) / f64::from(width)))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut normalized = stream.alloc_zeros::<f32>(elements as usize).unwrap();
            let mut expected = stream.alloc_zeros::<f32>(elements as usize).unwrap();
            let mut actual = stream.alloc_zeros::<f32>(elements as usize).unwrap();
            let eps = 1e-6f32;
            norm(&stream, &rms, &x, &scales, &mut normalized, width, eps).unwrap();
            for position in [0u32, 1, 127, 4095, 131071, 262143] {
                let state = stream.clone_htod(&[0u32, position]).unwrap();
                let mut launch = stream.launch_builder(&rope);
                launch
                    .arg(&normalized)
                    .arg(&freq)
                    .arg(&mut expected)
                    .arg(&width)
                    .arg(&elements)
                    .arg(&position);
                // SAFETY: three complete heads, bounded pair-wise output, independent buffers.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: ((elements / 2).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();
                for dynamic in [false, true] {
                    let kernel = module
                        .load_function(if dynamic {
                            "norm_rope_dynamic"
                        } else {
                            "norm_rope"
                        })
                        .unwrap();
                    let mut launch = stream.launch_builder(&kernel);
                    launch
                        .arg(&x)
                        .arg(&scales)
                        .arg(&freq)
                        .arg(&mut actual)
                        .arg(&width)
                        .arg(&eps);
                    if dynamic {
                        launch.arg(&state);
                    } else {
                        launch.arg(&position);
                    }
                    // SAFETY: one 256-thread block per validated 256/512-channel head; state has position at index 1.
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (3, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .unwrap();
                    let a = stream.clone_dtoh(&actual).unwrap();
                    let b = stream.clone_dtoh(&expected).unwrap();
                    stream.synchronize().unwrap();
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "width={width}, position={position}, dynamic={dynamic}"
                    );
                    let grouped = module
                        .load_function(if dynamic {
                            "prepare_qkv_dynamic"
                        } else {
                            "prepare_qkv"
                        })
                        .unwrap();
                    let mut ko = stream.alloc_zeros::<f32>(elements as usize).unwrap();
                    let mut vo = stream.alloc_zeros::<f32>(elements as usize).unwrap();
                    let heads = 3u32;
                    let mut launch = stream.launch_builder(&grouped);
                    launch
                        .arg(&x)
                        .arg(&x)
                        .arg(&x)
                        .arg(&scales)
                        .arg(&scales)
                        .arg(&scales)
                        .arg(&freq)
                        .arg(&mut actual)
                        .arg(&mut ko)
                        .arg(&mut vo)
                        .arg(&width)
                        .arg(&heads)
                        .arg(&heads)
                        .arg(&eps)
                        .arg(&eps)
                        .arg(&eps);
                    if dynamic {
                        launch.arg(&state);
                    } else {
                        launch.arg(&position);
                    }
                    // SAFETY: nine complete heads across three read-only aliased inputs, distinct full-sized outputs, supported width.
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (9, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .unwrap();
                    let q = stream.clone_dtoh(&actual).unwrap();
                    let k = stream.clone_dtoh(&ko).unwrap();
                    let v = stream.clone_dtoh(&vo).unwrap();
                    let n = stream.clone_dtoh(&normalized).unwrap();
                    stream.synchronize().unwrap();
                    for result in [q, k] {
                        assert_eq!(
                            result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "grouped Q/K, width={width}, position={position}, dynamic={dynamic}"
                        );
                    }
                    assert_eq!(
                        v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        n.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "grouped V must not rotate"
                    );
                    if dynamic && position <= 127 {
                        let size = (position as usize + 2) * elements as usize;
                        let mut kc = stream.clone_htod(&vec![12345.0f32; size]).unwrap();
                        let mut vc = stream.clone_htod(&vec![12345.0f32; size]).unwrap();
                        let kernel = module.load_function("prepare_qkv_cache_dynamic").unwrap();
                        let mut launch = stream.launch_builder(&kernel);
                        launch
                            .arg(&x)
                            .arg(&x)
                            .arg(&x)
                            .arg(&scales)
                            .arg(&scales)
                            .arg(&scales)
                            .arg(&freq)
                            .arg(&mut actual)
                            .arg(&mut kc)
                            .arg(&mut vc)
                            .arg(&width)
                            .arg(&heads)
                            .arg(&heads)
                            .arg(&eps)
                            .arg(&eps)
                            .arg(&eps)
                            .arg(&state);
                        // SAFETY: cache has position+2 complete slots; read-only aliases and disjoint outputs; one block per supported head.
                        unsafe {
                            launch.launch(LaunchConfig {
                                grid_dim: (9, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                        }
                        .unwrap();
                        let keys = stream.clone_dtoh(&kc).unwrap();
                        let values = stream.clone_dtoh(&vc).unwrap();
                        let q = stream.clone_dtoh(&actual).unwrap();
                        stream.synchronize().unwrap();
                        assert_eq!(
                            q.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                        let offset = position as usize * elements as usize;
                        for (cache, target) in [(keys, &b), (values, &n)] {
                            assert!(
                                cache[..offset]
                                    .iter()
                                    .chain(cache[offset + elements as usize..].iter())
                                    .all(|&v| v == 12345.0)
                            );
                            assert_eq!(
                                cache[offset..offset + elements as usize]
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                target.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
    }
    use super::*;
    use adaforge_domain::{AttentionConfig, AttentionOutput, Ffn, Q4Matrix, Rope};
    struct Fixture {
        norm: RmsNorm,
        q: Q4Matrix,
        k: Q4Matrix,
        v: Q4Matrix,
        qnorm: RmsNorm,
        knorm: RmsNorm,
        rope: Rope,
        attention: AttentionConfig,
        output: AttentionOutput,
        ffn: Ffn,
    }
    fn matrix(columns: u32, rows: u32, seed: u32) -> Q4Matrix {
        let mut bytes = Vec::new();
        for i in 0..columns / 32 * rows {
            bytes.extend(0x2000_u16.to_le_bytes());
            bytes
                .extend((0..16).map(|j| {
                    ((i + j + seed) % 16) as u8 | (((i + 15 - j + seed) % 16) as u8) << 4
                }));
        }
        Q4Matrix::new(columns, rows, bytes).unwrap()
    }
    impl Fixture {
        fn new(seed: u32, window: Option<u32>) -> Self {
            let norm = || RmsNorm::new(vec![0.75; 32], 1e-6).unwrap();
            Self {
                norm: norm(),
                q: matrix(32, 64, seed),
                k: matrix(32, 32, seed + 1),
                v: matrix(32, 32, seed + 2),
                qnorm: norm(),
                knorm: RmsNorm::new(vec![1.25; 32], 1e-6).unwrap(),
                rope: Rope::new(32, 10000.0, vec![1.0; 16], 3).unwrap(),
                attention: AttentionConfig::new(2, 1, 32, window, 1.0).unwrap(),
                output: AttentionOutput::new(matrix(64, 32, seed + 3), norm()).unwrap(),
                ffn: Ffn::new(
                    norm(),
                    matrix(32, 64, seed + 4),
                    matrix(32, 64, seed + 5),
                    matrix(64, 32, seed + 6),
                    norm(),
                    0.5,
                )
                .unwrap(),
            }
        }
        fn weights(&self) -> LayerWeights<'_> {
            LayerWeights {
                input_norm: &self.norm,
                matrices: [&self.q, &self.k, &self.v],
                head_norms: [&self.qnorm, &self.knorm],
                rope: &self.rope,
                attention: self.attention,
                output: &self.output,
                ffn: &self.ffn,
            }
        }
    }
    #[test]
    fn rejects_incompatible_width_and_token_capacity() {
        let f = Fixture::new(0, Some(2));
        let w = f.weights();
        assert!(validate(&w, 32, 3).is_ok());
        for (width, tokens) in [(64, 3), (32, 0), (32, 4), (32, 33)] {
            assert!(validate(&w, width, tokens).is_err());
        }
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn resident_blocks_keep_distinct_caches_and_reject_invalid_positions() {
        let a = Fixture::new(0, Some(2));
        let mut b = Fixture::new(3, None);
        // Change head geometry and share K/V, as at the local/global boundary.
        b.q = matrix(32, 128, 3);
        b.k = matrix(32, 64, 4);
        b.qnorm = RmsNorm::new(vec![0.75; 64], 1e-6).unwrap();
        b.knorm = RmsNorm::new(vec![1.25; 64], 1e-6).unwrap();
        let mut factors = vec![1.0; 32];
        factors[8..].fill(1e30);
        b.rope = Rope::new(64, 1e6, factors, 3).unwrap();
        b.attention = AttentionConfig::new(2, 1, 64, None, 1.0).unwrap();
        b.output = AttentionOutput::new(
            matrix(128, 32, 6),
            RmsNorm::new(vec![0.75; 32], 1e-6).unwrap(),
        )
        .unwrap();
        let wa = a.weights();
        let wb = LayerWeights {
            matrices: [&b.q, &b.k, &b.k],
            ..b.weights()
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut first = ResidentLayer::new(stream.clone(), &wa, 3).unwrap();
        let mut second = ResidentLayer::new(stream.clone(), &wb, 3).unwrap();
        let mut ca = ReferenceCache::default();
        let mut cb = ReferenceCache::default();
        let wrong = stream.alloc_zeros::<f32>(31).unwrap();
        assert!(first.run(&wrong, 0).is_err());
        for position in 0..3 {
            let input = (0..32)
                .map(|i| (i as f32 - 13.0) * 0.03125 + position as f32 * 0.125)
                .collect::<Vec<_>>();
            let device = stream.clone_htod(&input).unwrap();
            if position == 0 {
                assert!(first.run(&device, 1).is_err());
            }
            first.run(&device, position).unwrap();
            second.run(first.output(), position).unwrap();
            let expected_a = reference(&wa, &input, &mut ca, position).unwrap();
            let expected_b = reference(&wb, &fp32(expected_a.clone()), &mut cb, position).unwrap();
            super::super::matvec::compare(&expected_a, &first.ffn.snapshot().unwrap()).unwrap();
            super::super::matvec::compare(&expected_b, &second.ffn.snapshot().unwrap()).unwrap();
            let (ka, va) = first.cache.snapshot().unwrap();
            let (kb, vb) = second.cache.snapshot().unwrap();
            assert_eq!(ka.len(), (position as usize + 1) * 32);
            assert_eq!(kb.len(), (position as usize + 1) * 64);
            assert_ne!(ka, kb);
            assert_ne!(va, vb);
        }
        let before = second.cache.snapshot().unwrap();
        assert!(second.run(first.output(), 3).is_err());
        assert_eq!(second.cache.snapshot().unwrap(), before);
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn persistent_session_prefix_reset_stop_and_budget_are_consistent() {
        use adaforge_application::GenerationSession;
        let norm = || RmsNorm::new(vec![1.0; 256], 1e-6).unwrap();
        let f = Fixture {
            norm: norm(),
            q: matrix(256, 512, 0),
            k: matrix(256, 256, 1),
            v: matrix(256, 256, 2),
            qnorm: norm(),
            knorm: norm(),
            rope: Rope::new(256, 10000.0, vec![1.0; 128], 8).unwrap(),
            attention: AttentionConfig::new(2, 1, 256, None, 1.0).unwrap(),
            output: AttentionOutput::new(matrix(512, 256, 3), norm()).unwrap(),
            ffn: Ffn::new(
                norm(),
                matrix(256, 512, 4),
                matrix(256, 512, 5),
                matrix(512, 256, 6),
                norm(),
                1.0,
            )
            .unwrap(),
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let table = adaforge_domain::Q6Matrix::new(256, 5, vec![0; 210 * 5]).unwrap();
        let norm = norm();
        let mut session = CudaGenerationSession {
            graph: None,
            state: stream.alloc_zeros::<u32>(2).unwrap(),
            layers: vec![
                ResidentLayer::new(stream.clone(), &f.weights(), 8).unwrap(),
                ResidentLayer::new(stream.clone(), &f.weights(), 8).unwrap(),
            ],
            head: super::super::logits::ResidentHead::new(
                stream.clone(),
                adaforge_application::OutputHead {
                    norm: &norm,
                    matrix: &table,
                    cap: 30.0,
                },
            )
            .unwrap(),
            input: stream.alloc_zeros::<f32>(256).unwrap(),
            cached: Vec::new(),
            capacity: 8,
            vocabulary: 5,
            poisoned: false,
        };
        let mut emitted = Vec::new();
        let first = session
            .generate_stream(&[1, 2], &[4], 2, &mut |id| emitted.push(id))
            .unwrap();
        assert_eq!(emitted, first.ids);
        let stats = first.stats.as_ref().unwrap();
        assert_eq!(stats.prompt_tokens, 2);
        assert_eq!(stats.cached_tokens, 0);
        assert_eq!(stats.decode_tokens, 1);
        assert!(stats.prefill_ms >= 0.0 && stats.decode_ms >= 0.0);
        assert!(stats.total_ms >= stats.first_token_ms.unwrap() + stats.decode_ms);

        assert_eq!(first.ids, vec![0, 0]);
        assert!(!first.stopped);
        assert_eq!(session.cached, vec![1, 2, 0, 0]);
        assert!(session.generate(&[1, 3], &[4], 1).is_err());
        assert!(!session.poisoned);
        assert_eq!(session.cached.len(), 4);
        let second = session.generate(&[1, 2, 0, 0, 3], &[4], 1).unwrap();
        assert_eq!(second.ids, vec![0]);
        assert_eq!(session.cached, vec![1, 2, 0, 0, 3, 0]);
        for layer in &session.layers {
            assert_eq!(layer.next_position, 6);
            assert_eq!(layer.cache.resident().unwrap().tokens, 6);
        }
        assert!(session.generate(&[1, 2, 0, 0, 3, 0], &[4], 3).is_err());
        assert_eq!(session.cached.len(), 6);
        session.reset().unwrap();
        assert!(session.cached.is_empty());
        for layer in &session.layers {
            assert_eq!(layer.next_position, 0);
            assert_eq!(layer.cache.resident().unwrap().tokens, 0);
        }
        let stopped = session.generate(&[3], &[0], 2).unwrap();
        assert!(stopped.stopped);
        assert!(stopped.ids.is_empty());
        let stats = stopped.stats.as_ref().unwrap();
        assert!(stats.first_token_ms.is_none());
        assert_eq!(stats.decode_tokens, 0);
        assert_eq!(stats.decode_ms, 0.0);

        assert_eq!(session.cached, vec![3]);
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn graph_matches_direct_for_changing_tokens_positions_and_reset() {
        let norm = || RmsNorm::new(vec![1.0; 256], 1e-6).unwrap();
        let f = Fixture {
            norm: norm(),
            q: matrix(256, 512, 0),
            k: matrix(256, 256, 1),
            v: matrix(256, 256, 2),
            qnorm: norm(),
            knorm: norm(),
            rope: Rope::new(256, 10000.0, vec![1.0; 128], 4).unwrap(),
            attention: AttentionConfig::new(2, 1, 256, Some(2), 1.0).unwrap(),
            output: AttentionOutput::new(matrix(512, 256, 3), norm()).unwrap(),
            ffn: Ffn::new(
                norm(),
                matrix(256, 512, 4),
                matrix(256, 512, 5),
                matrix(512, 256, 6),
                norm(),
                1.0,
            )
            .unwrap(),
        };
        let ctx = CudaContext::new(0).unwrap();
        // SAFETY: both test sessions use this one stream; no cross-stream accesses exist.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().unwrap();
        let mut packed = Vec::new();
        for row in 0..5 {
            let mut b = vec![0; 210];
            for (i, x) in b[..192].iter_mut().enumerate() {
                *x = (i * 17 + row * 29) as u8;
            }
            b[192..208].fill(3);
            b[208..].copy_from_slice(&0x2800_u16.to_le_bytes());
            packed.extend(b);
        }
        let table = adaforge_domain::Q6Matrix::new(256, 5, packed).unwrap();
        let norm = norm();
        let make = || CudaGenerationSession {
            graph: None,
            state: stream.alloc_zeros::<u32>(2).unwrap(),
            layers: vec![
                ResidentLayer::new(stream.clone(), &f.weights(), 4).unwrap(),
                ResidentLayer::new(stream.clone(), &f.weights(), 4).unwrap(),
            ],
            head: super::super::logits::ResidentHead::new(
                stream.clone(),
                adaforge_application::OutputHead {
                    norm: &norm,
                    matrix: &table,
                    cap: 30.0,
                },
            )
            .unwrap(),
            input: stream.alloc_zeros::<f32>(256).unwrap(),
            cached: Vec::new(),
            capacity: 4,
            vocabulary: 5,
            poisoned: false,
        };
        let mut direct = make();
        let mut graph = make();
        graph.capture_graph().unwrap();
        for sequence in [[0, 3, 1, 2], [4, 2, 0, 1]] {
            for id in sequence {
                direct.consume(id).unwrap();
                graph.consume(id).unwrap();
                for (a, b) in direct.layers.iter().zip(&graph.layers) {
                    assert_eq!(a.ffn.snapshot().unwrap(), b.ffn.snapshot().unwrap());
                    assert_eq!(a.cache.snapshot().unwrap(), b.cache.snapshot().unwrap());
                }
            }
            use adaforge_application::GenerationSession;
            direct.reset().unwrap();
            graph.reset().unwrap();
        }
    }
}
