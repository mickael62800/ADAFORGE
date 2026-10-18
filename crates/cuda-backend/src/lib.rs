//! Backend CUDA : toutes les ressources GPU restent dans cet adaptateur.

use adaforge_application::{DiagnosticError, GpuCheckReport, GpuDiagnostics};
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

mod attention;
mod attention_output;
mod batch;
mod embedding;
mod ffn;
mod greedy;
#[cfg(test)]
mod half_conversion;
mod kv_cache;
mod logits;
mod matvec;
mod projection;
mod rms_norm;
mod stack;

pub struct CudaDiagnostics;

fn failure(stage: &str, error: impl std::fmt::Debug) -> DiagnosticError {
    DiagnosticError(format!("{stage}: {error:?}"))
}

impl GpuDiagnostics for CudaDiagnostics {
    fn open_session(
        &self,
        weights: &[adaforge_application::LayerWeights<'_>],
        head: adaforge_application::OutputHead<'_>,
        capacity: u32,
    ) -> Result<Box<dyn adaforge_application::GenerationSession>, DiagnosticError> {
        stack::open_session(weights, head, capacity)
    }

    fn generate(
        &self,
        weights: &[adaforge_application::LayerWeights<'_>],
        head: adaforge_application::OutputHead<'_>,
        prompt: &[u32],
        stop: &[u32],
        max_new: u32,
    ) -> Result<adaforge_application::GenerationTokens, DiagnosticError> {
        stack::generate(weights, head, prompt, stop, max_new)
    }

    fn check_logits(
        &self,
        weights: &[adaforge_application::LayerWeights<'_>],
        embeddings: &adaforge_domain::EmbeddingBatch,
        head: adaforge_application::OutputHead<'_>,
    ) -> Result<adaforge_application::StackReport, DiagnosticError> {
        stack::check_with_head(weights, embeddings, Some(head))
    }

    fn check_layer_stack(
        &self,
        weights: &[adaforge_application::LayerWeights<'_>],
        embeddings: &adaforge_domain::EmbeddingBatch,
    ) -> Result<adaforge_application::StackReport, DiagnosticError> {
        stack::check(weights, embeddings)
    }

    fn check_embedded_layer(
        &self,
        weights: adaforge_application::LayerWeights<'_>,
        embeddings: &adaforge_domain::EmbeddingBatch,
    ) -> Result<adaforge_application::RopeReport, DiagnosticError> {
        projection::check_embedded_layer(weights, embeddings)
    }
    fn check_embeddings(
        &self,
        batch: &adaforge_domain::EmbeddingBatch,
    ) -> Result<adaforge_application::EmbeddingReport, DiagnosticError> {
        embedding::check(batch)
    }

    fn check_layer(
        &self,
        weights: adaforge_application::LayerWeights<'_>,
    ) -> Result<adaforge_application::RopeReport, DiagnosticError> {
        projection::check_layer(weights)
    }
    fn check_attention_branch(
        &self,
        norm: &adaforge_domain::RmsNorm,
        matrices: [&adaforge_domain::Q4Matrix; 3],
        heads: [&adaforge_domain::RmsNorm; 2],
        rope: &adaforge_domain::Rope,
        config: adaforge_domain::AttentionConfig,
        branch: &adaforge_domain::AttentionOutput,
    ) -> Result<adaforge_application::RopeReport, DiagnosticError> {
        projection::check_attention_branch(norm, matrices, heads, rope, config, branch)
    }
    fn check_attention(
        &self,
        norm: &adaforge_domain::RmsNorm,
        matrices: [&adaforge_domain::Q4Matrix; 3],
        heads: [&adaforge_domain::RmsNorm; 2],
        rope: &adaforge_domain::Rope,
        config: adaforge_domain::AttentionConfig,
    ) -> Result<adaforge_application::RopeReport, DiagnosticError> {
        projection::check_attention(norm, matrices, heads, rope, config)
    }
    fn check_cached_kv(
        &self,
        norm: &adaforge_domain::RmsNorm,
        matrices: [&adaforge_domain::Q4Matrix; 3],
        heads: [&adaforge_domain::RmsNorm; 2],
        rope: &adaforge_domain::Rope,
    ) -> Result<adaforge_application::RopeReport, DiagnosticError> {
        projection::check_cached_kv(norm, matrices, heads, rope)
    }
    fn check_kv_cache(
        &self,
        layout: adaforge_domain::KvLayout,
    ) -> Result<adaforge_application::KvCacheReport, DiagnosticError> {
        kv_cache::check(layout)
    }
    fn check_vnorm(
        &self,
        norm: &adaforge_domain::RmsNorm,
        matrices: [&adaforge_domain::Q4Matrix; 3],
        heads: [&adaforge_domain::RmsNorm; 2],
        rope: &adaforge_domain::Rope,
    ) -> Result<adaforge_application::RopeReport, DiagnosticError> {
        projection::check_vnorm(norm, matrices, heads, rope)
    }
    fn check_rope(
        &self,
        norm: &adaforge_domain::RmsNorm,
        matrices: [&adaforge_domain::Q4Matrix; 3],
        heads: [&adaforge_domain::RmsNorm; 2],
        rope: &adaforge_domain::Rope,
    ) -> Result<adaforge_application::RopeReport, DiagnosticError> {
        projection::check_rope(norm, matrices, heads, rope)
    }
    fn check_qk_norm(
        &self,
        norm: &adaforge_domain::RmsNorm,
        matrices: [&adaforge_domain::Q4Matrix; 3],
        heads: [&adaforge_domain::RmsNorm; 2],
    ) -> Result<adaforge_application::QkNormReport, DiagnosticError> {
        projection::check_qk_norm(norm, matrices, heads)
    }
    fn check_qkv(
        &self,
        norm: &adaforge_domain::RmsNorm,
        matrices: [&adaforge_domain::Q4Matrix; 3],
    ) -> Result<adaforge_application::QkvReport, DiagnosticError> {
        projection::check_many(norm, &matrices)
    }
    fn check_projection(
        &self,
        norm: &adaforge_domain::RmsNorm,
        matrix: &adaforge_domain::Q4Matrix,
    ) -> Result<adaforge_application::MatvecReport, DiagnosticError> {
        projection::check(norm, matrix)
    }
    fn check_rms_norm(
        &self,
        norm: &adaforge_domain::RmsNorm,
    ) -> Result<adaforge_application::RmsNormReport, DiagnosticError> {
        rms_norm::check(norm)
    }
    fn check_matvec(
        &self,
        matrix: &adaforge_domain::Q4Matrix,
    ) -> Result<adaforge_application::MatvecReport, DiagnosticError> {
        matvec::check(matrix)
    }
    fn check(&self) -> Result<GpuCheckReport, DiagnosticError> {
        let ctx = CudaContext::new(0).map_err(|e| failure("Initialisation GPU 0", e))?;
        let device_name = ctx.name().map_err(|e| failure("Nom du GPU", e))?;
        let major = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| failure("Capacité CUDA", e))?;
        let minor = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| failure("Capacité CUDA", e))?;
        if (major, minor) != (8, 9) || !device_name.contains("RTX 4090") {
            return Err(DiagnosticError(format!(
                "Cible attendue : RTX 4090 sm_89 ; détectée : {device_name} sm_{major}{minor}"
            )));
        }
        let (free_memory_bytes, total_memory_bytes) = ctx
            .mem_get_info()
            .map_err(|e| failure("Mémoire disponible", e))?;
        let ptx = compile_ptx_with_opts(
            include_str!("../../../kernels/ada/vector_add.cu"),
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| failure("Compilation NVRTC (CUDA Toolkit requis)", e))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| failure("Chargement PTX", e))?;
        let kernel = module
            .load_function("vector_add")
            .map_err(|e| failure("Chargement kernel", e))?;
        let stream = ctx.default_stream();
        // Taille non multiple de 256 pour exercer la protection du dernier bloc.
        let n = 1_000_003_u32;
        let a: Vec<f32> = (0..n).map(|i| ((i % 1024) as f32 - 512.0) * 0.25).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 127) as f32 - 63.0) * 0.5).collect();
        let a_gpu = stream
            .clone_htod(&a)
            .map_err(|e| failure("Transfert A", e))?;
        let b_gpu = stream
            .clone_htod(&b)
            .map_err(|e| failure("Transfert B", e))?;
        let mut output = stream
            .alloc_zeros::<f32>(n as usize)
            .map_err(|e| failure("Allocation sortie", e))?;
        let config = LaunchConfig {
            grid_dim: (n.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // Échauffement : exclure la première exécution du temps mesuré.
        let mut warmup = stream.launch_builder(&kernel);
        warmup.arg(&a_gpu).arg(&b_gpu).arg(&mut output).arg(&n);
        // SAFETY: signature identique au kernel, trois buffers f32 de n éléments,
        // entrée en lecture seule, sortie exclusive, bornage i < n dans le kernel.
        unsafe { warmup.launch(config) }.map_err(|e| failure("Échauffement kernel", e))?;
        stream
            .synchronize()
            .map_err(|e| failure("Synchronisation échauffement", e))?;
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| failure("Événement début", e))?;
        let mut launch = stream.launch_builder(&kernel);
        launch.arg(&a_gpu).arg(&b_gpu).arg(&mut output).arg(&n);
        // SAFETY: mêmes signature, tailles, exclusivité et bornage que l’échauffement.
        // Toutes les ressources restent vivantes jusqu’à la synchronisation du stream.
        unsafe { launch.launch(config) }.map_err(|e| failure("Lancement kernel", e))?;
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| failure("Événement fin", e))?;
        end.synchronize()
            .map_err(|e| failure("Synchronisation kernel", e))?;
        let kernel_ms = start
            .elapsed_ms(&end)
            .map_err(|e| failure("Chronométrage CUDA", e))?;
        let actual = stream
            .clone_dtoh(&output)
            .map_err(|e| failure("Retour CPU", e))?;
        stream
            .synchronize()
            .map_err(|e| failure("Synchronisation transfert", e))?;
        let max_absolute_error = validate(&a, &b, &actual)?;
        // Les buffers, événements, module, stream et contexte sont libérés par RAII.
        Ok(GpuCheckReport {
            device_name,
            compute_capability: (major, minor),
            free_memory_bytes,
            total_memory_bytes,
            checked_elements: actual.len(),
            max_absolute_error,
            kernel_ms,
        })
    }
}

fn validate(a: &[f32], b: &[f32], actual: &[f32]) -> Result<f32, DiagnosticError> {
    if a.len() != b.len() || a.len() != actual.len() {
        return Err(DiagnosticError(
            "Tailles incohérentes lors de la validation".into(),
        ));
    }
    let mut max_error = 0.0_f32;
    for (i, ((a, b), value)) in a.iter().zip(b).zip(actual).enumerate() {
        let expected = a + b;
        let error = (value - expected).abs();
        if !value.is_finite() || !expected.is_finite() || error > 1e-6 {
            return Err(DiagnosticError(format!(
                "Résultat incorrect à l’indice {i}: attendu {expected}, obtenu {value}"
            )));
        }
        max_error = max_error.max(error);
    }
    Ok(max_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_wrong_values_non_finite_and_lengths() {
        for actual in [vec![4.0], vec![f32::NAN], vec![f32::INFINITY], vec![]] {
            assert!(validate(&[1.0], &[2.0], &actual).is_err());
        }
        assert_eq!(
            validate(&[-1.0, 1.25], &[2.0, -0.5], &[1.0, 0.75]).unwrap(),
            0.0
        );
    }
    #[test]
    #[ignore = "Nécessite une RTX 4090, le pilote NVIDIA et NVRTC CUDA 13.3"]
    fn real_gpu_check() {
        let report = CudaDiagnostics.check().expect("Diagnostic réel GPU");
        assert_eq!(report.checked_elements, 1_000_003);
        assert_eq!(report.max_absolute_error, 0.0);
        assert!(report.kernel_ms.is_finite() && report.kernel_ms >= 0.0);
    }
}
#[cfg(test)] mod bw_probe_test;
