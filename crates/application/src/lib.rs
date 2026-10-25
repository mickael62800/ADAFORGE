//! Cas d’usage et ports indépendants de CUDA et du transport.

use std::{error::Error, fmt};

/// Résultat d’un diagnostic matériel et numérique, sans types CUDA.
#[derive(Debug)]
pub struct GpuCheckReport {
    pub device_name: String,
    pub compute_capability: (i32, i32),
    pub free_memory_bytes: usize,
    pub total_memory_bytes: usize,
    pub checked_elements: usize,
    pub max_absolute_error: f32,
    pub kernel_ms: f32,
}

#[derive(Debug)]
pub struct DiagnosticError(pub String);

impl fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for DiagnosticError {}

/// Port sortant : le backend réalise un diagnostic complet.
pub trait GpuDiagnostics {
    fn open_session(
        &self,
        _weights: &[LayerWeights<'_>],
        _head: OutputHead<'_>,
        _capacity: u32,
    ) -> Result<Box<dyn GenerationSession>, DiagnosticError> {
        Err(DiagnosticError("Session persistante indisponible".into()))
    }

    fn generate(
        &self,
        _weights: &[LayerWeights<'_>],
        _head: OutputHead<'_>,
        _prompt: &[u32],
        _stop: &[u32],
        _max_new: u32,
    ) -> Result<GenerationTokens, DiagnosticError> {
        Err(DiagnosticError("Génération indisponible".into()))
    }

    fn check_logits(
        &self,
        _weights: &[LayerWeights<'_>],
        _embeddings: &adaforge_domain::EmbeddingBatch,
        _head: OutputHead<'_>,
    ) -> Result<StackReport, DiagnosticError> {
        Err(DiagnosticError("Diagnostic logits indisponible".into()))
    }

    fn check_layer_stack(
        &self,
        _weights: &[LayerWeights<'_>],
        _embeddings: &adaforge_domain::EmbeddingBatch,
    ) -> Result<StackReport, DiagnosticError> {
        Err(DiagnosticError(
            "Diagnostic multi-couche indisponible".into(),
        ))
    }

    fn check_embedded_layer(
        &self,
        _weights: LayerWeights<'_>,
        _embeddings: &adaforge_domain::EmbeddingBatch,
    ) -> Result<RopeReport, DiagnosticError> {
        Err(DiagnosticError(
            "Diagnostic embeddings/couche indisponible".into(),
        ))
    }
    fn check_embeddings(
        &self,
        _batch: &adaforge_domain::EmbeddingBatch,
    ) -> Result<EmbeddingReport, DiagnosticError> {
        Err(DiagnosticError("Diagnostic embeddings indisponible".into()))
    }
    fn check_layer(&self, _weights: LayerWeights<'_>) -> Result<RopeReport, DiagnosticError> {
        Err(DiagnosticError(
            "Diagnostic couche dense indisponible".into(),
        ))
    }
    fn check_attention_branch(
        &self,
        _norm: &adaforge_domain::RmsNorm,
        _matrices: [&adaforge_domain::Q4Matrix; 3],
        _heads: [&adaforge_domain::RmsNorm; 2],
        _rope: &adaforge_domain::Rope,
        _config: adaforge_domain::AttentionConfig,
        _branch: &adaforge_domain::AttentionOutput,
    ) -> Result<RopeReport, DiagnosticError> {
        Err(DiagnosticError(
            "Diagnostic branche attention indisponible".into(),
        ))
    }
    fn check_attention(
        &self,
        _norm: &adaforge_domain::RmsNorm,
        _matrices: [&adaforge_domain::Q4Matrix; 3],
        _heads: [&adaforge_domain::RmsNorm; 2],
        _rope: &adaforge_domain::Rope,
        _config: adaforge_domain::AttentionConfig,
    ) -> Result<RopeReport, DiagnosticError> {
        Err(DiagnosticError("Diagnostic attention indisponible".into()))
    }
    fn check_cached_kv(
        &self,
        _norm: &adaforge_domain::RmsNorm,
        _matrices: [&adaforge_domain::Q4Matrix; 3],
        _heads: [&adaforge_domain::RmsNorm; 2],
        _rope: &adaforge_domain::Rope,
    ) -> Result<RopeReport, DiagnosticError> {
        Err(DiagnosticError(
            "Diagnostic chaîne avec cache indisponible".into(),
        ))
    }
    fn check_kv_cache(
        &self,
        _layout: adaforge_domain::KvLayout,
    ) -> Result<KvCacheReport, DiagnosticError> {
        Err(DiagnosticError("Diagnostic cache KV indisponible".into()))
    }
    fn check_vnorm(
        &self,
        _norm: &adaforge_domain::RmsNorm,
        _matrices: [&adaforge_domain::Q4Matrix; 3],
        _heads: [&adaforge_domain::RmsNorm; 2],
        _rope: &adaforge_domain::Rope,
    ) -> Result<RopeReport, DiagnosticError> {
        Err(DiagnosticError(
            "Diagnostic normalisation V indisponible".into(),
        ))
    }
    fn check_rope(
        &self,
        _norm: &adaforge_domain::RmsNorm,
        _matrices: [&adaforge_domain::Q4Matrix; 3],
        _heads: [&adaforge_domain::RmsNorm; 2],
        _rope: &adaforge_domain::Rope,
    ) -> Result<RopeReport, DiagnosticError> {
        Err(DiagnosticError("Diagnostic RoPE indisponible".into()))
    }
    fn check_qk_norm(
        &self,
        _norm: &adaforge_domain::RmsNorm,
        _matrices: [&adaforge_domain::Q4Matrix; 3],
        _heads: [&adaforge_domain::RmsNorm; 2],
    ) -> Result<QkNormReport, DiagnosticError> {
        Err(DiagnosticError(
            "Diagnostic normalisation Q/K indisponible".into(),
        ))
    }
    fn check_qkv(
        &self,
        _norm: &adaforge_domain::RmsNorm,
        _matrices: [&adaforge_domain::Q4Matrix; 3],
    ) -> Result<QkvReport, DiagnosticError> {
        Err(DiagnosticError("Diagnostic QKV indisponible".into()))
    }
    fn check_projection(
        &self,
        _norm: &adaforge_domain::RmsNorm,
        _matrix: &adaforge_domain::Q4Matrix,
    ) -> Result<MatvecReport, DiagnosticError> {
        Err(DiagnosticError("Diagnostic projection indisponible".into()))
    }
    fn check_rms_norm(
        &self,
        _norm: &adaforge_domain::RmsNorm,
    ) -> Result<RmsNormReport, DiagnosticError> {
        Err(DiagnosticError("Diagnostic RMSNorm indisponible".into()))
    }
    fn check(&self) -> Result<GpuCheckReport, DiagnosticError>;
    fn check_matvec(
        &self,
        _matrix: &adaforge_domain::Q4Matrix,
    ) -> Result<MatvecReport, DiagnosticError> {
        Err(DiagnosticError(
            "Diagnostic matrice-vecteur indisponible".into(),
        ))
    }
}

pub fn check_gpu(backend: &impl GpuDiagnostics) -> Result<GpuCheckReport, DiagnosticError> {
    backend.check()
}

/// Valeurs consultables sans conserver le vocabulaire complet en mémoire.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Array(Vec<MetadataValue>),
    ArraySummary { element_type: u32, length: u64 },
}
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub format_version: u32,
    pub file_bytes: u64,
    pub data_offset: u64,
    pub metadata: std::collections::BTreeMap<String, MetadataValue>,
    pub tensors: Vec<adaforge_domain::TensorInfo>,
}
pub trait ModelInspector {
    fn read_q6_matrix(
        &self,
        _path: &std::path::Path,
        _name: &str,
    ) -> Result<adaforge_domain::Q6Matrix, DiagnosticError> {
        Err(DiagnosticError("Lecture Q6_K indisponible".into()))
    }

    fn read_tokenizer(
        &self,
        _path: &std::path::Path,
    ) -> Result<adaforge_domain::GemmaTokenizer, DiagnosticError> {
        Err(DiagnosticError("Lecture tokenizer indisponible".into()))
    }
    fn read_embeddings(
        &self,
        _path: &std::path::Path,
        _ids: &[u32],
    ) -> Result<adaforge_domain::EmbeddingBatch, DiagnosticError> {
        Err(DiagnosticError("Lecture embeddings indisponible".into()))
    }
    fn read_f32_vector(
        &self,
        _path: &std::path::Path,
        _name: &str,
    ) -> Result<Vec<f32>, DiagnosticError> {
        Err(DiagnosticError("Lecture vecteur F32 indisponible".into()))
    }
    fn read_rms_norm(
        &self,
        _path: &std::path::Path,
        _name: &str,
    ) -> Result<adaforge_domain::RmsNorm, DiagnosticError> {
        Err(DiagnosticError("Lecture RMSNorm indisponible".into()))
    }
    fn read_q4_matrix(
        &self,
        _path: &std::path::Path,
        _name: &str,
    ) -> Result<adaforge_domain::Q4Matrix, DiagnosticError> {
        Err(DiagnosticError("Lecture Q4_0 indisponible".into()))
    }
    fn inspect(&self, path: &std::path::Path) -> Result<ModelInfo, DiagnosticError>;
}
pub fn inspect_model(
    inspector: &impl ModelInspector,
    path: &std::path::Path,
) -> Result<ModelInfo, DiagnosticError> {
    inspector.inspect(path)
}

#[derive(Debug)]
pub struct MatvecReport {
    pub rows: u32,
    pub columns: u32,
    pub checked_vectors: usize,
    pub max_absolute_error: f64,
    /// Max(error / (atol + rtol * abs(reference))). Doit être <= 1.
    pub max_tolerance_ratio: f64,
    pub median_kernel_ms: f32,
    pub repetitions: usize,
}
pub fn check_model_matvec(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    name: &str,
) -> Result<MatvecReport, DiagnosticError> {
    let matrix = inspector.read_q4_matrix(path, name)?;
    backend.check_matvec(&matrix)
}

#[derive(Debug)]
pub struct RmsNormReport {
    pub width: usize,
    pub vectors: usize,
    pub epsilon: f32,
    pub max_absolute_error: f64,
    pub max_tolerance_ratio: f64,
    pub median_kernel_ms: f32,
}
pub fn check_model_rms_norm(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    name: &str,
) -> Result<RmsNormReport, DiagnosticError> {
    let norm = inspector.read_rms_norm(path, name)?;
    backend.check_rms_norm(&norm)
}

/// Enchaîne normalisation et projection ; les détails GPU restent dans le backend.
pub fn check_model_projection(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
) -> Result<MatvecReport, DiagnosticError> {
    let norm = inspector.read_rms_norm(path, "blk.0.attn_norm.weight")?;
    let matrix = inspector.read_q4_matrix(path, "blk.0.attn_q.weight")?;
    if norm.weights().len() != matrix.columns() as usize {
        return Err(DiagnosticError(
            "Largeur RMSNorm incompatible avec la projection Q".into(),
        ));
    }
    backend.check_projection(&norm, &matrix)
}

#[derive(Debug)]
pub struct ProjectionError {
    pub rows: u32,
    pub max_absolute_error: f64,
    pub max_tolerance_ratio: f64,
}
#[derive(Debug)]
pub struct QkvReport {
    pub embedding_ids: Option<Vec<u32>>,
    pub ffn: Option<OutputBranchReport>,
    pub output_branch: Option<OutputBranchReport>,
    pub attention: Option<AttentionCheckReport>,
    pub cache: Option<CachedPrefixReport>,
    pub value_uses_key_weights: bool,
    pub columns: u32,
    pub checked_vectors: usize,
    pub projections: Vec<ProjectionError>,
    pub median_kernel_ms: f32,
    pub repetitions: usize,
}
struct LoadedQkv {
    norm: adaforge_domain::RmsNorm,
    q: adaforge_domain::Q4Matrix,
    k: adaforge_domain::Q4Matrix,
    value: Option<adaforge_domain::Q4Matrix>,
}
impl LoadedQkv {
    fn matrices(&self) -> [&adaforge_domain::Q4Matrix; 3] {
        [&self.q, &self.k, self.value.as_ref().unwrap_or(&self.k)]
    }
}
fn load_qkv(
    inspector: &impl ModelInspector,
    path: &std::path::Path,
    layer: u32,
) -> Result<LoadedQkv, DiagnosticError> {
    let info = inspector.inspect(path)?;
    if !matches!(info.metadata.get("general.architecture"),Some(MetadataValue::Text(a)) if a=="gemma4")
    {
        return Err(DiagnosticError(
            "Diagnostic QKV modèle réservé à Gemma 4".into(),
        ));
    }
    match info.metadata.get("gemma4.block_count") {
        Some(MetadataValue::Unsigned(n)) if u64::from(layer) < *n => {}
        _ => return Err(DiagnosticError("Indice de couche hors limites".into())),
    }
    let norm = inspector.read_rms_norm(path, &format!("blk.{layer}.attn_norm.weight"))?;
    let q = inspector.read_q4_matrix(path, &format!("blk.{layer}.attn_q.weight"))?;
    let k = inspector.read_q4_matrix(path, &format!("blk.{layer}.attn_k.weight"))?;
    let value_name = format!("blk.{layer}.attn_v.weight");
    // Gemma 4 alternative attention : en l’absence de V, la projection brute
    // réutilise les poids K. Les normalisations Q/K ultérieures sont distinctes.
    let value = if info.tensors.iter().any(|t| t.name == value_name) {
        Some(inspector.read_q4_matrix(path, &value_name)?)
    } else {
        None
    };
    let v = value.as_ref().unwrap_or(&k);
    if k.rows() != v.rows() {
        return Err(DiagnosticError("Dimensions K/V incompatibles".into()));
    }
    Ok(LoadedQkv { norm, q, k, value })
}
pub fn check_model_qkv(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
) -> Result<QkvReport, DiagnosticError> {
    let loaded = load_qkv(inspector, path, layer)?;
    let mut report = backend.check_qkv(&loaded.norm, loaded.matrices())?;
    report.value_uses_key_weights = loaded.value.is_none();
    Ok(report)
}

#[cfg(test)]
mod qkv_tests {
    use super::*;
    use adaforge_domain::{Q4Matrix, RmsNorm, TensorInfo, WeightType};
    struct Model {
        has_v: bool,
    }
    impl ModelInspector for Model {
        fn inspect(&self, _: &std::path::Path) -> Result<ModelInfo, DiagnosticError> {
            let mut metadata = std::collections::BTreeMap::new();
            metadata.insert(
                "general.architecture".into(),
                MetadataValue::Text("gemma4".into()),
            );
            metadata.insert("gemma4.block_count".into(), MetadataValue::Unsigned(1));
            let tensors = if self.has_v {
                vec![TensorInfo {
                    name: "blk.0.attn_v.weight".into(),
                    dimensions: vec![32, 1],
                    weight_type: WeightType::Q4_0,
                    file_offset: 0,
                    byte_length: 18,
                }]
            } else {
                vec![]
            };
            Ok(ModelInfo {
                format_version: 3,
                file_bytes: 0,
                data_offset: 0,
                metadata,
                tensors,
            })
        }
        fn read_rms_norm(&self, _: &std::path::Path, _: &str) -> Result<RmsNorm, DiagnosticError> {
            Ok(RmsNorm::new(vec![1.0; 32], 1e-6).unwrap())
        }
        fn read_q4_matrix(
            &self,
            _: &std::path::Path,
            name: &str,
        ) -> Result<Q4Matrix, DiagnosticError> {
            assert!(self.has_v || !name.ends_with("attn_v.weight"));
            Ok(Q4Matrix::new(32, 1, vec![0; 18]).unwrap())
        }
    }
    struct Backend {
        expect_alias: bool,
    }
    impl GpuDiagnostics for Backend {
        fn check(&self) -> Result<GpuCheckReport, DiagnosticError> {
            panic!("GPU check inattendu")
        }
        fn check_qkv(
            &self,
            _: &RmsNorm,
            matrices: [&Q4Matrix; 3],
        ) -> Result<QkvReport, DiagnosticError> {
            assert_eq!(std::ptr::eq(matrices[1], matrices[2]), self.expect_alias);
            Ok(QkvReport {
                cache: None,
                attention: None,
                output_branch: None,
                ffn: None,
                embedding_ids: None,
                value_uses_key_weights: false,
                columns: 32,
                checked_vectors: 4,
                projections: vec![],
                median_kernel_ms: 0.0,
                repetitions: 10,
            })
        }
    }
    #[test]
    fn absent_v_reuses_k_and_present_v_is_loaded_separately() {
        for has_v in [false, true] {
            let report = check_model_qkv(
                &Model { has_v },
                &Backend {
                    expect_alias: !has_v,
                },
                std::path::Path::new("unused"),
                0,
            )
            .unwrap();
            assert_eq!(report.value_uses_key_weights, !has_v);
        }
    }
    #[test]
    fn out_of_range_layer_is_rejected() {
        assert!(
            check_model_qkv(
                &Model { has_v: false },
                &Backend { expect_alias: true },
                std::path::Path::new("unused"),
                1
            )
            .is_err()
        );
    }
}

#[derive(Debug)]
pub struct QkNormReport {
    pub qkv: QkvReport,
    pub head_width: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
}
pub fn check_model_qk_norm(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
) -> Result<QkNormReport, DiagnosticError> {
    let loaded = load_qkv(inspector, path, layer)?;
    let q_norm = inspector.read_rms_norm(path, &format!("blk.{layer}.attn_q_norm.weight"))?;
    let k_norm = inspector.read_rms_norm(path, &format!("blk.{layer}.attn_k_norm.weight"))?;
    let mut report = backend.check_qk_norm(&loaded.norm, loaded.matrices(), [&q_norm, &k_norm])?;
    report.qkv.value_uses_key_weights = loaded.value.is_none();
    Ok(report)
}

#[derive(Debug)]
pub struct RopeReport {
    pub heads: QkNormReport,
    pub positions: Vec<u32>,
    pub base: f64,
    pub attenuated_pairs: usize,
    pub local_attention: bool,
}
pub fn check_model_rope(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
) -> Result<RopeReport, DiagnosticError> {
    check_model_positioned(
        inspector,
        backend,
        path,
        layer,
        false,
        false,
        AttentionStage::None,
    )
}
pub fn check_model_vnorm(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
) -> Result<RopeReport, DiagnosticError> {
    check_model_positioned(
        inspector,
        backend,
        path,
        layer,
        true,
        false,
        AttentionStage::None,
    )
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum AttentionStage<'a> {
    None,
    Raw,
    Output,
    Layer,
    Embedded(&'a [u32]),
}
fn check_model_positioned(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
    normalize_value: bool,
    cache_prefix: bool,
    attention_stage: AttentionStage<'_>,
) -> Result<RopeReport, DiagnosticError> {
    if attention_stage == AttentionStage::Layer
        || matches!(attention_stage, AttentionStage::Embedded(_))
    {
        let dense = stack::load(inspector, path, layer)?;
        let mut report = if let AttentionStage::Embedded(ids) = attention_stage {
            if layer != 0 {
                return Err(DiagnosticError(
                    "Les embeddings alimentent uniquement le premier bloc dans ce diagnostic"
                        .into(),
                ));
            }
            let embeddings = inspector.read_embeddings(path, ids)?;
            backend.check_embedded_layer(dense.weights(), &embeddings)?
        } else {
            backend.check_layer(dense.weights())?
        };
        report.local_attention = dense.local;
        report.heads.qkv.value_uses_key_weights = dense.loaded.value.is_none();
        return Ok(report);
    }
    let loaded = load_qkv(inspector, path, layer)?;
    let qnorm = inspector.read_rms_norm(path, &format!("blk.{layer}.attn_q_norm.weight"))?;
    let knorm = inspector.read_rms_norm(path, &format!("blk.{layer}.attn_k_norm.weight"))?;
    let info = inspector.inspect(path)?;
    let (rope, local) = load_rope(inspector, path, &info, layer, qnorm.weights().len() as u32)?;
    let mut report = if attention_stage != AttentionStage::None {
        let width = rope.width();
        let window = if local {
            match info.metadata.get("gemma4.attention.sliding_window") {
                Some(MetadataValue::Unsigned(v)) => Some(
                    u32::try_from(*v)
                        .map_err(|_| DiagnosticError("Fenêtre attention invalide".into()))?,
                ),
                _ => return Err(DiagnosticError("Fenêtre attention absente".into())),
            }
        } else {
            None
        };
        if !loaded.q.rows().is_multiple_of(width) || !loaded.k.rows().is_multiple_of(width) {
            return Err(DiagnosticError("Dimensions attention incompatibles".into()));
        }
        let config = adaforge_domain::AttentionConfig::new(
            loaded.q.rows() / width,
            loaded.k.rows() / width,
            width,
            window,
            1.0,
        )
        .map_err(DiagnosticError)?;
        if attention_stage == AttentionStage::Output {
            let matrix =
                inspector.read_q4_matrix(path, &format!("blk.{layer}.attn_output.weight"))?;
            let post_norm = inspector
                .read_rms_norm(path, &format!("blk.{layer}.post_attention_norm.weight"))?;
            let branch = adaforge_domain::AttentionOutput::new(matrix, post_norm)
                .map_err(DiagnosticError)?;
            backend.check_attention_branch(
                &loaded.norm,
                loaded.matrices(),
                [&qnorm, &knorm],
                &rope,
                config,
                &branch,
            )?
        } else {
            backend.check_attention(
                &loaded.norm,
                loaded.matrices(),
                [&qnorm, &knorm],
                &rope,
                config,
            )?
        }
    } else if cache_prefix {
        backend.check_cached_kv(&loaded.norm, loaded.matrices(), [&qnorm, &knorm], &rope)?
    } else if normalize_value {
        backend.check_vnorm(&loaded.norm, loaded.matrices(), [&qnorm, &knorm], &rope)?
    } else {
        backend.check_rope(&loaded.norm, loaded.matrices(), [&qnorm, &knorm], &rope)?
    };
    report.heads.qkv.value_uses_key_weights = loaded.value.is_none();
    report.local_attention = local;
    Ok(report)
}
fn load_rope(
    inspector: &impl ModelInspector,
    path: &std::path::Path,
    info: &ModelInfo,
    layer: u32,
    width: u32,
) -> Result<(adaforge_domain::Rope, bool), DiagnosticError> {
    let bad = || DiagnosticError("Métadonnées RoPE absentes ou incompatibles".into());
    let local = match info.metadata.get("gemma4.attention.sliding_window_pattern") {
        Some(MetadataValue::Array(values)) => match values.get(layer as usize) {
            Some(MetadataValue::Bool(b)) => *b,
            _ => return Err(bad()),
        },
        _ => return Err(bad()),
    };
    let suffix = if local { "_swa" } else { "" };
    let base = match info.metadata.get(&format!("gemma4.rope.freq_base{suffix}")) {
        Some(MetadataValue::Float(v)) => *v,
        _ => return Err(bad()),
    };
    let dimension = match info
        .metadata
        .get(&format!("gemma4.rope.dimension_count{suffix}"))
    {
        Some(MetadataValue::Unsigned(v)) => *v,
        _ => return Err(bad()),
    };
    if dimension != u64::from(width) {
        return Err(DiagnosticError("RoPE partiel non pris en charge : dimension_count doit égaler la largeur de tête ; les facteurs proportionnels GGUF sont pris en charge".into()));
    }
    let context = match info.metadata.get("gemma4.context_length") {
        Some(MetadataValue::Unsigned(v)) => u32::try_from(*v).map_err(|_| bad())?,
        _ => return Err(bad()),
    };
    let factors = if local {
        vec![1.0; width as usize / 2]
    } else {
        let name = format!("blk.{layer}.rope_freqs.weight");
        let name = if info.tensors.iter().any(|t| t.name == name) {
            name.as_str()
        } else {
            "rope_freqs.weight"
        };
        inspector.read_f32_vector(path, name)?
    };
    let rope =
        adaforge_domain::Rope::new(width, base, factors, context).map_err(DiagnosticError)?;
    Ok((rope, local))
}

#[cfg(test)]
mod rope_tests {
    use super::*;
    struct Factors;
    impl ModelInspector for Factors {
        fn inspect(&self, _: &std::path::Path) -> Result<ModelInfo, DiagnosticError> {
            panic!("Pas d’inspection attendue")
        }
        fn read_f32_vector(
            &self,
            _: &std::path::Path,
            name: &str,
        ) -> Result<Vec<f32>, DiagnosticError> {
            assert_eq!(name, "rope_freqs.weight");
            Ok(vec![1.0, 1e30])
        }
    }
    fn info() -> ModelInfo {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert(
            "gemma4.attention.sliding_window_pattern".into(),
            MetadataValue::Array(vec![MetadataValue::Bool(true), MetadataValue::Bool(false)]),
        );
        for (key, value) in [
            ("gemma4.rope.freq_base_swa", 10000.0),
            ("gemma4.rope.freq_base", 1000000.0),
        ] {
            metadata.insert(key.into(), MetadataValue::Float(value));
        }
        for (key, value) in [
            ("gemma4.rope.dimension_count_swa", 4),
            ("gemma4.rope.dimension_count", 4),
            ("gemma4.context_length", 262144),
        ] {
            metadata.insert(key.into(), MetadataValue::Unsigned(value));
        }
        ModelInfo {
            format_version: 3,
            file_bytes: 0,
            data_offset: 0,
            metadata,
            tensors: vec![],
        }
    }
    #[test]
    fn selects_local_base_and_global_factors_and_rejects_missing_metadata() {
        let path = std::path::Path::new("unused");
        let mut info = info();
        let (local, is_local) = load_rope(&Factors, path, &info, 0, 4).unwrap();
        assert!(is_local);
        assert_eq!(local.base(), 10000.0);
        assert_eq!(local.factors(), &[1.0, 1.0]);
        let (global, is_local) = load_rope(&Factors, path, &info, 1, 4).unwrap();
        assert!(!is_local);
        assert_eq!(global.base(), 1000000.0);
        assert_eq!(global.factors(), &[1.0, 1e30]);
        assert!(load_rope(&Factors, path, &info, 2, 4).is_err());
        assert!(load_rope(&Factors, path, &info, 0, 8).is_err());
        info.metadata.remove("gemma4.rope.freq_base_swa");
        assert!(load_rope(&Factors, path, &info, 0, 4).is_err());
    }
}

#[derive(Debug)]
pub struct KvCacheReport {
    pub capacity: u32,
    pub heads: u32,
    pub width: u32,
    pub allocated_bytes: u64,
    pub verified_values: usize,
    pub overflow_rejected: bool,
    pub reset_verified: bool,
}
pub fn check_model_kv_cache(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
) -> Result<KvCacheReport, DiagnosticError> {
    let info = inspector.inspect(path)?;
    if !matches!(info.metadata.get("general.architecture"),Some(MetadataValue::Text(a)) if a=="gemma4")
    {
        return Err(DiagnosticError(
            "Diagnostic cache modèle réservé à Gemma 4".into(),
        ));
    }
    let error = || DiagnosticError("Géométrie KV absente ou invalide".into());
    let blocks = match info.metadata.get("gemma4.block_count") {
        Some(MetadataValue::Unsigned(v)) => *v,
        _ => return Err(error()),
    };
    if u64::from(layer) >= blocks {
        return Err(DiagnosticError("Indice de couche hors limites".into()));
    }
    let local = match info.metadata.get("gemma4.attention.sliding_window_pattern") {
        Some(MetadataValue::Array(v)) => match v.get(layer as usize) {
            Some(MetadataValue::Bool(b)) => *b,
            _ => return Err(error()),
        },
        _ => return Err(error()),
    };
    let to_u32 = |v: &MetadataValue| -> Result<u32, DiagnosticError> {
        match v {
            MetadataValue::Unsigned(n) => u32::try_from(*n).map_err(|_| error()),
            MetadataValue::Signed(n) => u32::try_from(*n).map_err(|_| error()),
            _ => Err(error()),
        }
    };
    let heads = match info.metadata.get("gemma4.attention.head_count_kv") {
        Some(MetadataValue::Array(v)) => to_u32(v.get(layer as usize).ok_or_else(error)?)?,
        Some(v) => to_u32(v)?,
        _ => return Err(error()),
    };
    let key = if local {
        "gemma4.attention.key_length_swa"
    } else {
        "gemma4.attention.key_length"
    };
    let value = if local {
        "gemma4.attention.value_length_swa"
    } else {
        "gemma4.attention.value_length"
    };
    let width = to_u32(info.metadata.get(key).ok_or_else(error)?)?;
    if width != to_u32(info.metadata.get(value).ok_or_else(error)?)? {
        return Err(DiagnosticError(
            "Largeurs K/V différentes non prises en charge".into(),
        ));
    }
    let layout = adaforge_domain::KvLayout::new(8, heads, width).map_err(DiagnosticError)?;
    backend.check_kv_cache(layout)
}

#[derive(Debug)]
pub struct CachedPrefixReport {
    pub tokens: u32,
    pub verified_values: usize,
    pub max_absolute_error: f64,
    pub max_tolerance_ratio: f64,
    pub previous_tokens_unchanged: bool,
    pub overflow_rejected: bool,
}
pub fn check_model_cached_kv(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
) -> Result<RopeReport, DiagnosticError> {
    check_model_positioned(
        inspector,
        backend,
        path,
        layer,
        true,
        true,
        AttentionStage::None,
    )
}

#[derive(Debug)]
pub struct AttentionCheckReport {
    pub checked_values: usize,
    pub max_absolute_error: f64,
    pub max_tolerance_ratio: f64,
    pub window: Option<u32>,
    pub scale: f32,
}
pub fn check_model_attention(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
) -> Result<RopeReport, DiagnosticError> {
    check_model_positioned(
        inspector,
        backend,
        path,
        layer,
        true,
        true,
        AttentionStage::Raw,
    )
}

#[derive(Debug)]
pub struct OutputBranchReport {
    pub checked_values: usize,
    pub width: u32,
    pub max_absolute_error: f64,
    pub max_tolerance_ratio: f64,
}
pub fn check_model_attention_branch(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
) -> Result<RopeReport, DiagnosticError> {
    check_model_positioned(
        inspector,
        backend,
        path,
        layer,
        true,
        true,
        AttentionStage::Output,
    )
}

pub struct LayerWeights<'a> {
    pub input_norm: &'a adaforge_domain::RmsNorm,
    pub matrices: [&'a adaforge_domain::Q4Matrix; 3],
    pub head_norms: [&'a adaforge_domain::RmsNorm; 2],
    pub rope: &'a adaforge_domain::Rope,
    pub attention: adaforge_domain::AttentionConfig,
    pub output: &'a adaforge_domain::AttentionOutput,
    pub ffn: &'a adaforge_domain::Ffn,
}
pub fn check_model_layer(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    layer: u32,
) -> Result<RopeReport, DiagnosticError> {
    check_model_positioned(
        inspector,
        backend,
        path,
        layer,
        true,
        true,
        AttentionStage::Layer,
    )
}

#[derive(Debug)]
pub struct EmbeddingReport {
    pub width: u32,
    pub vocabulary: u32,
    pub token_ids: Vec<u32>,
    pub scale: f32,
    pub max_absolute_error: f64,
    pub max_tolerance_ratio: f64,
    pub verified_values: usize,
}
pub fn check_model_embeddings(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    ids: &[u32],
) -> Result<EmbeddingReport, DiagnosticError> {
    let batch = inspector.read_embeddings(path, ids)?;
    backend.check_embeddings(&batch)
}

pub fn check_model_embedded_layer(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
) -> Result<RopeReport, DiagnosticError> {
    check_model_positioned(
        inspector,
        backend,
        path,
        0,
        true,
        true,
        AttentionStage::Embedded(&[]),
    )
}

pub struct TokenizationReport {
    pub ids: Vec<u32>,
    pub decoded: String,
    pub vocabulary: usize,
}
pub fn tokenize_text(
    inspector: &impl ModelInspector,
    path: &std::path::Path,
    text: &str,
    options: adaforge_domain::TokenizeOptions,
) -> Result<TokenizationReport, DiagnosticError> {
    let tokenizer = inspector.read_tokenizer(path)?;
    let ids = tokenizer.encode(text, options).map_err(DiagnosticError)?;
    let decoded = tokenizer
        .decode(&ids[usize::from(options.add_bos)..], true)
        .map_err(DiagnosticError)?;
    Ok(TokenizationReport {
        ids,
        decoded,
        vocabulary: tokenizer.vocabulary(),
    })
}

pub use adaforge_domain::TokenizeOptions;

pub struct TextLayerReport {
    pub tokenization: TokenizationReport,
    pub layer: RopeReport,
}
/// Diagnostic borné : texte brut, BOS, embeddings puis bloc zéro.
pub fn check_text_layer(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    text: &str,
) -> Result<TextLayerReport, DiagnosticError> {
    let tokenization = tokenize_text(
        inspector,
        path,
        text,
        TokenizeOptions {
            add_bos: true,
            parse_special: false,
        },
    )?;
    if tokenization.ids.is_empty() || tokenization.ids.len() > 32 {
        return Err(DiagnosticError(
            "Diagnostic texte limité à 32 tokens, BOS inclus ; aucun texte n’est tronqué".into(),
        ));
    }
    let layer = check_model_positioned(
        inspector,
        backend,
        path,
        0,
        true,
        true,
        AttentionStage::Embedded(&tokenization.ids),
    )?;
    Ok(TextLayerReport {
        tokenization,
        layer,
    })
}

mod stack;
pub use stack::{LayerGeometry, StackReport, TextStackReport, check_text_stack};

pub struct OutputHead<'a> {
    pub norm: &'a adaforge_domain::RmsNorm,
    pub matrix: &'a adaforge_domain::Q6Matrix,
    pub cap: f32,
}
pub struct LogitsReport {
    pub vocabulary: u32,
    pub position: u32,
    pub norm_error: f64,
    pub norm_ratio: f64,
    pub raw_error: f64,
    pub raw_ratio: f64,
    pub capped_error: f64,
    pub capped_ratio: f64,
}

pub use stack::check_text_logits;

#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub prefill_ms: f64,
    pub first_token_ms: Option<f64>,
    pub decode_tokens: usize,
    pub decode_ms: f64,
    pub total_ms: f64,
}
pub struct GenerationTokens {
    pub ids: Vec<u32>,
    pub stopped: bool,
    pub stats: Option<GenerationStats>,
}
pub struct GenerationReport {
    pub prompt_ids: Vec<u32>,
    pub generated_ids: Vec<u32>,
    pub text: String,
    pub stopped: bool,
    pub stats: Option<GenerationStats>,
}
pub use stack::{ask_text, generate_text};

/// Port de session : les poids et caches appartiennent exclusivement au backend.
pub trait GenerationSession {
    fn generate_stream(
        &mut self,
        prompt: &[u32],
        stop: &[u32],
        max_new: u32,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<GenerationTokens, DiagnosticError> {
        let result = self.generate(prompt, stop, max_new)?;
        for &id in &result.ids {
            on_token(id);
        }
        Ok(result)
    }

    fn generate(
        &mut self,
        prompt: &[u32],
        stop: &[u32],
        max_new: u32,
    ) -> Result<GenerationTokens, DiagnosticError>;
    fn reset(&mut self) -> Result<(), DiagnosticError>;
    /// Teacher-forced scoring (validation of approximate kernels): after the
    /// prompt, feeds `continuation` token by token and returns, for every
    /// next-token prediction, the log-normalizer of the softcapped logits and
    /// the `top_k` best (id, softcapped logit). Leaves the session reset.
    fn score(
        &mut self,
        _prompt: &[u32],
        _continuation: &[u32],
        _top_k: usize,
    ) -> Result<Vec<ScoredPosition>, DiagnosticError> {
        Err(DiagnosticError("Scoring non disponible pour ce backend".into()))
    }
}
/// One teacher-forced prediction: log-sum-exp of all softcapped logits and
/// the best (id, softcapped logit) pairs, highest first.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredPosition {
    pub log_normalizer: f64,
    pub top: Vec<(u32, f32)>,
}
pub use stack::{CHAT_CONTEXT, ChatSession, open_chat};
