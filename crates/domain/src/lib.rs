//! Description des tenseurs, indépendante des formats de fichiers et de CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WeightType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q6K,
    Bf16,
}
impl WeightType {
    pub fn block_layout(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q8_0 => (32, 34),
            Self::Q6K => (256, 210),
        }
    }
}
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Ordre de stockage : dimension contiguë en premier.
    pub dimensions: Vec<u64>,
    pub weight_type: WeightType,
    pub file_offset: u64,
    pub byte_length: u64,
}

mod q4;
pub use q4::Q4Matrix;

mod rms_norm;
pub use rms_norm::RmsNorm;

mod rope;
pub use rope::Rope;

mod kv_cache;
pub use kv_cache::{KvLayout, KvState};

mod attention;
pub use attention::AttentionConfig;

mod attention_output;
pub use attention_output::AttentionOutput;

mod ffn;
pub use ffn::Ffn;

mod embedding;
pub use embedding::EmbeddingBatch;

mod tokenizer;
pub use tokenizer::{GemmaTokenizer, TokenizeOptions};

mod q6;
pub use q6::{Q6Matrix, softcap};

mod sampling;
pub use sampling::greedy;
