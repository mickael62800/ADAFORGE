use super::*;

pub(super) struct DenseLayer {
    pub(super) loaded: LoadedQkv,
    pub(super) local: bool,
    qnorm: adaforge_domain::RmsNorm,
    knorm: adaforge_domain::RmsNorm,
    rope: adaforge_domain::Rope,
    attention: adaforge_domain::AttentionConfig,
    output: adaforge_domain::AttentionOutput,
    ffn: adaforge_domain::Ffn,
}
impl DenseLayer {
    pub(super) fn weights(&self) -> LayerWeights<'_> {
        LayerWeights {
            input_norm: &self.loaded.norm,
            matrices: self.loaded.matrices(),
            head_norms: [&self.qnorm, &self.knorm],
            rope: &self.rope,
            attention: self.attention,
            output: &self.output,
            ffn: &self.ffn,
        }
    }
}
pub(super) fn load(
    inspector: &impl ModelInspector,
    path: &std::path::Path,
    layer: u32,
) -> Result<DenseLayer, DiagnosticError> {
    let loaded = load_qkv(inspector, path, layer)?;
    let qnorm = inspector.read_rms_norm(path, &format!("blk.{layer}.attn_q_norm.weight"))?;
    let knorm = inspector.read_rms_norm(path, &format!("blk.{layer}.attn_k_norm.weight"))?;
    let info = inspector.inspect(path)?;
    let (rope, local) = load_rope(inspector, path, &info, layer, qnorm.weights().len() as u32)?;
    let width = rope.width();
    let window = if local {
        match info.metadata.get("gemma4.attention.sliding_window") {
            Some(MetadataValue::Unsigned(n)) => {
                Some(u32::try_from(*n).map_err(|_| DiagnosticError("Fenêtre invalide".into()))?)
            }
            _ => return Err(DiagnosticError("Fenêtre absente".into())),
        }
    } else {
        None
    };
    if !loaded.q.rows().is_multiple_of(width) || !loaded.k.rows().is_multiple_of(width) {
        return Err(DiagnosticError("Dimensions de têtes incompatibles".into()));
    }
    let attention = adaforge_domain::AttentionConfig::new(
        loaded.q.rows() / width,
        loaded.k.rows() / width,
        width,
        window,
        1.0,
    )
    .map_err(DiagnosticError)?;
    let output = adaforge_domain::AttentionOutput::new(
        inspector.read_q4_matrix(path, &format!("blk.{layer}.attn_output.weight"))?,
        inspector.read_rms_norm(path, &format!("blk.{layer}.post_attention_norm.weight"))?,
    )
    .map_err(DiagnosticError)?;
    // Dense Gemma path shared by isolated and consecutive block diagnostics.
    if info
        .tensors
        .iter()
        .any(|t| t.name.starts_with(&format!("blk.{layer}.ffn_gate_inp")))
        || !matches!(
            info.metadata.get("gemma4.embedding_length_per_layer_input"),
            Some(MetadataValue::Unsigned(0))
        )
    {
        return Err(DiagnosticError(
            "Ce chemin dense ne prend pas en charge MoE ou embeddings par couche".into(),
        ));
    }
    let scale_name = format!("blk.{layer}.layer_output_scale.weight");
    let scale = if info.tensors.iter().any(|t| t.name == scale_name) {
        let v = inspector.read_f32_vector(path, &scale_name)?;
        if v.len() != 1 {
            return Err(DiagnosticError("Échelle de couche non scalaire".into()));
        }
        v[0]
    } else {
        1.0
    };
    let ffn = adaforge_domain::Ffn::new(
        inspector.read_rms_norm(path, &format!("blk.{layer}.ffn_norm.weight"))?,
        inspector.read_q4_matrix(path, &format!("blk.{layer}.ffn_gate.weight"))?,
        inspector.read_q4_matrix(path, &format!("blk.{layer}.ffn_up.weight"))?,
        inspector.read_q4_matrix(path, &format!("blk.{layer}.ffn_down.weight"))?,
        inspector.read_rms_norm(path, &format!("blk.{layer}.post_ffw_norm.weight"))?,
        scale,
    )
    .map_err(DiagnosticError)?;

    Ok(DenseLayer {
        loaded,
        local,
        qnorm,
        knorm,
        rope,
        attention,
        output,
        ffn,
    })
}

pub struct LayerGeometry {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_width: u32,
    pub window: Option<u32>,
    pub shared_kv_weights: bool,
    pub attenuated_pairs: usize,
}
pub struct StackReport {
    pub token_ids: Vec<u32>,
    pub width: u32,
    pub layers: Vec<OutputBranchReport>,
    pub geometry: Vec<LayerGeometry>,
    pub logits: Option<LogitsReport>,
}
pub struct TextStackReport {
    pub tokenization: TokenizationReport,
    pub stack: StackReport,
}
pub fn check_text_stack(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    text: &str,
    depth: u32,
) -> Result<TextStackReport, DiagnosticError> {
    if !(1..=48).contains(&depth) {
        return Err(DiagnosticError(
            "Diagnostic limité à 1 à 48 blocs consécutifs".into(),
        ));
    }
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
    let embeddings = inspector.read_embeddings(path, &tokenization.ids)?;
    let layers = (0..depth)
        .map(|i| load(inspector, path, i))
        .collect::<Result<Vec<_>, _>>()?;
    let weights = layers.iter().map(DenseLayer::weights).collect::<Vec<_>>();
    let stack = backend.check_layer_stack(&weights, &embeddings)?;
    Ok(TextStackReport {
        tokenization,
        stack,
    })
}

pub fn check_text_logits(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    text: &str,
) -> Result<TextStackReport, DiagnosticError> {
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
            "Diagnostic limité à 32 tokens, BOS inclus".into(),
        ));
    }
    let info = inspector.inspect(path)?;
    if !matches!(
        info.metadata.get("gemma4.block_count"),
        Some(MetadataValue::Unsigned(48))
    ) {
        return Err(DiagnosticError(
            "Ce diagnostic cible les 48 blocs Gemma 4".into(),
        ));
    }
    let cap = match info.metadata.get("gemma4.final_logit_softcapping") {
        Some(MetadataValue::Float(v)) if v.is_finite() && *v > 0.0 => *v as f32,
        _ => return Err(DiagnosticError("Softcapping absent ou invalide".into())),
    };
    let name = if info.tensors.iter().any(|t| t.name == "output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let matrix = inspector.read_q6_matrix(path, name)?;
    if matrix.rows() as usize != tokenization.vocabulary {
        return Err(DiagnosticError("Tête et tokenizer incompatibles".into()));
    }
    let norm = inspector.read_rms_norm(path, "output_norm.weight")?;
    let embeddings = inspector.read_embeddings(path, &tokenization.ids)?;
    let layers = (0..48)
        .map(|i| load(inspector, path, i))
        .collect::<Result<Vec<_>, _>>()?;
    let weights = layers.iter().map(DenseLayer::weights).collect::<Vec<_>>();
    let stack = backend.check_logits(
        &weights,
        &embeddings,
        OutputHead {
            norm: &norm,
            matrix: &matrix,
            cap,
        },
    )?;
    Ok(TextStackReport {
        tokenization,
        stack,
    })
}

/// Première complétion de texte brut, sans template de conversation.
fn generate_impl(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    text: &str,
    max_new: u32,
    chat: bool,
) -> Result<GenerationReport, DiagnosticError> {
    if !(1..=128).contains(&max_new) {
        return Err(DiagnosticError(
            "Génération limitée à 1 à 128 nouveaux tokens".into(),
        ));
    }
    let tokenizer = inspector.read_tokenizer(path)?;
    let prompt_ids = if chat {
        tokenizer.encode_user_turn(text)
    } else {
        tokenizer.encode(
            text,
            TokenizeOptions {
                add_bos: true,
                parse_special: false,
            },
        )
    }
    .map_err(DiagnosticError)?;
    if prompt_ids.is_empty() || prompt_ids.len() > 256 {
        return Err(DiagnosticError(
            "Prompt limité à 256 tokens, BOS inclus, sans troncature".into(),
        ));
    }
    let model = load_generation(inspector, path, &tokenizer)?;
    let weights = model
        .layers
        .iter()
        .map(DenseLayer::weights)
        .collect::<Vec<_>>();
    let stop = &model.stop;
    let result = backend.generate(
        &weights,
        OutputHead {
            norm: &model.norm,
            matrix: &model.matrix,
            cap: model.cap,
        },
        &prompt_ids,
        stop,
        max_new,
    )?;
    // A token limit may cut a UTF-8 byte sequence; render an incomplete suffix lossily.
    let decoded = tokenizer
        .decode_generated(&result.ids)
        .map_err(DiagnosticError)?;
    Ok(GenerationReport {
        prompt_ids,
        generated_ids: result.ids,
        text: decoded,
        stopped: result.stopped,
        stats: result.stats,
    })
}

pub fn generate_text(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    text: &str,
    max_new: u32,
) -> Result<GenerationReport, DiagnosticError> {
    generate_impl(inspector, backend, path, text, max_new, false)
}
pub fn ask_text(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    text: &str,
    max_new: u32,
) -> Result<GenerationReport, DiagnosticError> {
    generate_impl(inspector, backend, path, text, max_new, true)
}

struct GenerationModel {
    norm: adaforge_domain::RmsNorm,
    matrix: adaforge_domain::Q6Matrix,
    cap: f32,
    layers: Vec<DenseLayer>,
    stop: Vec<u32>,
}
fn load_generation(
    inspector: &impl ModelInspector,
    path: &std::path::Path,
    tokenizer: &adaforge_domain::GemmaTokenizer,
) -> Result<GenerationModel, DiagnosticError> {
    let info = inspector.inspect(path)?;
    if !matches!(
        info.metadata.get("gemma4.block_count"),
        Some(MetadataValue::Unsigned(48))
    ) || info.tensors.iter().any(|t| t.name == "output.weight")
    {
        return Err(DiagnosticError("Première génération réservée à Gemma 4 dense 48 blocs avec poids de sortie liés aux embeddings".into()));
    }
    let cap = match info.metadata.get("gemma4.final_logit_softcapping") {
        Some(MetadataValue::Float(v)) if v.is_finite() && *v > 0.0 => *v as f32,
        _ => return Err(DiagnosticError("Softcap invalide".into())),
    };
    let matrix = inspector.read_q6_matrix(path, "token_embd.weight")?;
    if matrix.rows() as usize != tokenizer.vocabulary() {
        return Err(DiagnosticError("Vocabulaire incompatible".into()));
    }
    let norm = inspector.read_rms_norm(path, "output_norm.weight")?;
    let layers = (0..48)
        .map(|i| load(inspector, path, i))
        .collect::<Result<Vec<_>, _>>()?;
    let stop = ["<eos>", "<turn|>", "<|tool_response>"]
        .iter()
        .filter_map(|s| tokenizer.token_id(s))
        .collect::<Vec<_>>();
    if stop.is_empty() {
        return Err(DiagnosticError("Marqueurs de fin absents".into()));
    }
    Ok(GenerationModel {
        norm,
        matrix,
        cap,
        layers,
        stop,
    })
}
pub struct ChatSession {
    tokenizer: adaforge_domain::GemmaTokenizer,
    backend: Box<dyn GenerationSession>,
    history: Vec<u32>,
    stop: Vec<u32>,
    max_new: u32,
    capacity: u32,
}
impl ChatSession {
    pub fn send(&mut self, message: &str) -> Result<GenerationReport, DiagnosticError> {
        self.send_stream(message, &mut |_| {})
    }
    pub fn send_stream(
        &mut self,
        message: &str,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<GenerationReport, DiagnosticError> {
        if message.trim().is_empty() {
            return Err(DiagnosticError("Message vide".into()));
        }
        let turn = self
            .tokenizer
            .encode_user_turn(message)
            .map_err(DiagnosticError)?;
        let mut prompt = self.history.clone();
        prompt.extend_from_slice(if prompt.is_empty() { &turn } else { &turn[1..] });
        if prompt.len() + self.max_new as usize > self.capacity as usize {
            return Err(DiagnosticError(format!("Contexte de {} tokens dépassé ; utiliser /reset ou un message plus court. Aucun historique tronqué.", self.capacity)));
        }
        let mut streamed = Vec::new();
        let mut emitted = 0;
        let mut decode_error = None;
        let result =
            self.backend
                .generate_stream(&prompt, &self.stop, self.max_new, &mut |id| {
                    streamed.push(id);
                    match self.tokenizer.decode_generated_prefix(&streamed) {
                        Ok(text) => {
                            if text.len() > emitted {
                                on_text(&text[emitted..]);
                                emitted = text.len();
                            }
                        }
                        Err(e) => decode_error = Some(e),
                    }
                })?;
        if let Some(error) = decode_error {
            return Err(DiagnosticError(error));
        }
        let text = self
            .tokenizer
            .decode_generated(&result.ids)
            .map_err(DiagnosticError)?;
        if text.len() > emitted {
            on_text(&text[emitted..]);
        }
        let closure = self
            .tokenizer
            .encode(
                "<turn|>\n",
                TokenizeOptions {
                    add_bos: false,
                    parse_special: true,
                },
            )
            .map_err(DiagnosticError)?;
        self.history = prompt.clone();
        self.history.extend_from_slice(&result.ids);
        self.history.extend(closure);
        Ok(GenerationReport {
            prompt_ids: prompt,
            generated_ids: result.ids,
            text,
            stopped: result.stopped,
            stats: result.stats,
        })
    }
    pub fn max_new_tokens(&self) -> u32 {
        self.max_new
    }
    pub fn set_max_new_tokens(&mut self, value: u32) -> Result<(), DiagnosticError> {
        if !(1..=128).contains(&value) {
            return Err(DiagnosticError("Budget limité à 1 à 128 tokens".into()));
        }
        if self.history.len() + value as usize > self.capacity as usize {
            return Err(DiagnosticError(
                "Contexte insuffisant pour ce budget ; utiliser /reset".into(),
            ));
        }
        self.max_new = value;
        Ok(())
    }
    pub fn reset(&mut self) -> Result<(), DiagnosticError> {
        self.backend.reset()?;
        self.history.clear();
        Ok(())
    }
}
pub fn open_chat(
    inspector: &impl ModelInspector,
    backend: &impl GpuDiagnostics,
    path: &std::path::Path,
    max_new: u32,
) -> Result<ChatSession, DiagnosticError> {
    if !(1..=128).contains(&max_new) {
        return Err(DiagnosticError(
            "Génération limitée à 1 à 128 nouveaux tokens".into(),
        ));
    }
    let diagnostics = std::env::var("ADAFORGE_DIAGNOSTICS").as_deref() == Ok("1");
    let started = std::time::Instant::now();
    let tokenizer = inspector.read_tokenizer(path)?;
    let model = load_generation(inspector, path, &tokenizer)?;
    if diagnostics {
        eprintln!("LOAD_PHASE {{\"file_and_parse_ms\":{:.0}}}", started.elapsed().as_secs_f64() * 1000.0);
    }
    let weights = model
        .layers
        .iter()
        .map(DenseLayer::weights)
        .collect::<Vec<_>>();
    let session = backend.open_session(
        &weights,
        OutputHead {
            norm: &model.norm,
            matrix: &model.matrix,
            cap: model.cap,
        },
        CHAT_CONTEXT,
    )?;
    Ok(ChatSession {
        tokenizer,
        backend: session,
        history: Vec::new(),
        stop: model.stop,
        max_new,
        capacity: CHAT_CONTEXT,
    })
}
/// Chat context window in tokens (prompt history + answer).
pub const CHAT_CONTEXT: u32 = 32768;

#[cfg(test)]
mod chat_tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};
    struct Fake {
        calls: Rc<RefCell<Vec<Vec<u32>>>>,
        ids: Vec<u32>,
    }
    impl GenerationSession for Fake {
        fn generate(
            &mut self,
            _prompt: &[u32],
            _stop: &[u32],
            _max_new: u32,
        ) -> Result<GenerationTokens, DiagnosticError> {
            panic!("Expected streaming port");
        }
        fn generate_stream(
            &mut self,
            prompt: &[u32],
            _stop: &[u32],
            _max_new: u32,
            on_token: &mut dyn FnMut(u32),
        ) -> Result<GenerationTokens, DiagnosticError> {
            self.calls.borrow_mut().push(prompt.to_vec());
            for &id in &self.ids {
                on_token(id);
            }
            Ok(GenerationTokens {
                ids: self.ids.clone(),
                stopped: true,
                stats: None,
            })
        }
        fn reset(&mut self) -> Result<(), DiagnosticError> {
            Ok(())
        }
    }
    #[test]
    fn chat_stream_preserves_utf8_history_and_budget_on_refusal() {
        let mut tokens = [
            "<bos>",
            "<eos>",
            "<|turn>",
            "<turn|>",
            "<|channel>",
            "<channel|>",
        ]
        .map(String::from)
        .to_vec();
        let mut types = vec![3; tokens.len()];
        for b in 0..=255 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(6);
        }
        let tokenizer = adaforge_domain::GemmaTokenizer::new(tokens, types, vec![], 0).unwrap();
        let ids = vec![
            tokenizer.token_id("<0xC3>").unwrap(),
            tokenizer.token_id("<0xA9>").unwrap(),
        ];
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut chat = ChatSession {
            tokenizer,
            backend: Box::new(Fake {
                calls: calls.clone(),
                ids,
            }),
            history: vec![],
            stop: vec![1],
            max_new: 3,
            capacity: 512,
        };
        let mut chunks = Vec::new();
        let result = chat
            .send_stream("salut", &mut |s| chunks.push(s.to_string()))
            .unwrap();
        assert_eq!(chunks, vec!["é"]);
        assert_eq!(result.text, "é");
        let history = chat.history.clone();
        assert!(chat.set_max_new_tokens(0).is_err());
        assert!(chat.set_max_new_tokens(129).is_err());
        assert_eq!(chat.max_new_tokens(), 3);
        assert!(chat.send(&"x".repeat(600)).is_err());
        assert_eq!(chat.history, history);
        assert_eq!(calls.borrow().len(), 1);
        chat.set_max_new_tokens(128).unwrap();
        assert_eq!(chat.history, history);
        chat.send("suite").unwrap();
        assert!(calls.borrow()[1].starts_with(&history));
    }
}
