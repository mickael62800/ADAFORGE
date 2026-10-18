//! Première chaîne GPU résidente : RMSNorm puis projection Q4_0.
use super::{failure, matvec};
use adaforge_application::{DiagnosticError, MatvecReport, ProjectionError, QkvReport};
use adaforge_domain::{Q4Matrix, RmsNorm};
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

#[derive(Default, Clone, Copy)]
struct ChainTail<'a> {
    attention: Option<adaforge_domain::AttentionConfig>,
    output: Option<&'a adaforge_domain::AttentionOutput>,
    ffn: Option<&'a adaforge_domain::Ffn>,
    embeddings: Option<&'a adaforge_domain::EmbeddingBatch>,
}
pub(crate) fn check(norm: &RmsNorm, matrix: &Q4Matrix) -> Result<MatvecReport, DiagnosticError> {
    let report = check_many(norm, &[matrix])?;
    let error = &report.projections[0];
    Ok(MatvecReport {
        rows: error.rows,
        columns: report.columns,
        checked_vectors: report.checked_vectors,
        max_absolute_error: error.max_absolute_error,
        max_tolerance_ratio: error.max_tolerance_ratio,
        median_kernel_ms: report.median_kernel_ms,
        repetitions: report.repetitions,
    })
}

pub(crate) fn check_many(
    norm: &RmsNorm,
    matrices: &[&Q4Matrix],
) -> Result<QkvReport, DiagnosticError> {
    check_chain(
        norm,
        matrices,
        None,
        None,
        false,
        false,
        ChainTail::default(),
    )
}

pub(crate) fn check_qk_norm(
    norm: &RmsNorm,
    matrices: [&Q4Matrix; 3],
    heads: [&RmsNorm; 2],
) -> Result<adaforge_application::QkNormReport, DiagnosticError> {
    validate_heads(&matrices, &heads)?;
    let report = check_chain(
        norm,
        &matrices,
        Some(heads),
        None,
        false,
        false,
        ChainTail::default(),
    )?;
    Ok(adaforge_application::QkNormReport {
        qkv: report,
        head_width: heads[0].weights().len() as u32,
        q_heads: matrices[0].rows() / heads[0].weights().len() as u32,
        kv_heads: matrices[1].rows() / heads[1].weights().len() as u32,
    })
}
fn validate_heads(matrices: &[&Q4Matrix; 3], heads: &[&RmsNorm; 2]) -> Result<(), DiagnosticError> {
    let width = heads[0].weights().len() as u32;
    if width != heads[1].weights().len() as u32
        || !matrices[0].rows().is_multiple_of(width)
        || !matrices[1].rows().is_multiple_of(width)
        || matrices[1].rows() != matrices[2].rows()
        || !(matrices[0].rows() / width).is_multiple_of(matrices[1].rows() / width)
    {
        return Err(DiagnosticError(
            "Dimensions des têtes Q/K incompatibles".into(),
        ));
    }
    Ok(())
}
fn check_chain(
    norm: &RmsNorm,
    matrices: &[&Q4Matrix],
    head_norms: Option<[&RmsNorm; 2]>,
    rope: Option<&adaforge_domain::Rope>,
    normalize_value: bool,
    cache_prefix: bool,
    tail: ChainTail<'_>,
) -> Result<QkvReport, DiagnosticError> {
    let attention = tail.attention;
    let cases = tail
        .embeddings
        .map_or(if rope.is_some() { 8 } else { 4 }, |b| b.token_ids().len());
    if matrices.is_empty()
        || matrices
            .iter()
            .any(|m| norm.weights().len() != m.columns() as usize)
    {
        return Err(DiagnosticError(
            "Dimensions incompatibles entre RMSNorm et projection".into(),
        ));
    }
    if cache_prefix
        && (matrices.len() != 3
            || !normalize_value
            || rope.is_none_or(|r| (r.context() as usize) < cases))
    {
        return Err(DiagnosticError("La chaîne cache exige trois projections, V normalisé et un contexte suffisant pour les tokens".into()));
    }
    if let Some(c) = attention
        && (!cache_prefix
            || matrices.len() != 3
            || u64::from(c.q_heads()) * u64::from(c.width()) != u64::from(matrices[0].rows())
            || u64::from(c.kv_heads()) * u64::from(c.width()) != u64::from(matrices[1].rows()))
    {
        return Err(DiagnosticError(
            "Géométrie attention incompatible avec les projections/cache".into(),
        ));
    }
    if let Some(branch) = tail.output
        && (attention.is_none()
            || branch.matrix().columns() != matrices[0].rows()
            || branch.matrix().rows() != matrices[0].columns())
    {
        return Err(DiagnosticError(
            "Dimensions de branche attention incompatibles".into(),
        ));
    }
    if let Some(ffn) = tail.ffn
        && (tail.output.is_none() || ffn.width() != matrices[0].columns())
    {
        return Err(DiagnosticError(
            "FFN incompatible avec la branche attention".into(),
        ));
    }
    if let Some(batch) = tail.embeddings
        && (tail.ffn.is_none()
            || batch.width() != matrices[0].columns()
            || batch.token_ids().is_empty()
            || batch.token_ids().len() > 32
            || !cache_prefix)
    {
        return Err(DiagnosticError("Le diagnostic embeddings/couche exige entre un et 32 IDs, une largeur compatible et une chaîne complète avec cache".into()));
    }
    let ctx = CudaContext::new(0).map_err(|e| failure("Initialisation projection", e))?;
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
    .map_err(|e| failure("Compilation chaîne", e))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| failure("Module chaîne", e))?;
    let rms = module
        .load_function("rms_norm")
        .map_err(|e| failure("Kernel RMSNorm", e))?;
    let project = module
        .load_function("q4_matvec")
        .map_err(|e| failure("Kernel projection", e))?;
    let rotary = module
        .load_function("rope_neox")
        .map_err(|e| failure("Kernel RoPE", e))?;
    let rms_unscaled = module
        .load_function("rms_norm_unscaled")
        .map_err(|e| failure("Kernel RMSNorm V", e))?;
    let stream = ctx.default_stream();
    let columns = matrices[0].columns();
    let epsilon = norm.epsilon();
    // Allocations et upload des poids une seule fois pour tout le diagnostic.
    let weights = matrices
        .iter()
        .map(|m| {
            stream
                .clone_htod(m.bytes())
                .map_err(|e| failure("Poids Q4_0", e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scales = stream
        .clone_htod(norm.weights())
        .map_err(|e| failure("Poids RMSNorm", e))?;
    let mut input = stream
        .alloc_zeros::<f32>(columns as usize)
        .map_err(|e| failure("Buffer entrée", e))?;
    let mut normalized = stream
        .alloc_zeros::<f32>(columns as usize)
        .map_err(|e| failure("Buffer intermédiaire", e))?;
    let mut outputs = matrices
        .iter()
        .map(|m| {
            stream
                .alloc_zeros::<f32>(m.rows() as usize)
                .map_err(|e| failure("Buffer sortie", e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Sorties Q/K distinctes : ne pas normaliser V par accident lorsque K et V
    // partagent leurs poids. Les projections brutes restent disponibles sur le GPU.
    let head_scales = if let Some(heads) = head_norms {
        heads
            .iter()
            .map(|h| {
                stream
                    .clone_htod(h.weights())
                    .map_err(|e| failure("Poids tête", e))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let mut head_outputs = if head_norms.is_some() {
        matrices[..2]
            .iter()
            .map(|m| {
                stream
                    .alloc_zeros::<f32>(m.rows() as usize)
                    .map_err(|e| failure("Buffer tête", e))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let inv_freq = rope
        .map(|r| {
            stream
                .clone_htod(r.inverse_frequencies())
                .map_err(|e| failure("Fréquences RoPE", e))
        })
        .transpose()?;
    let mut rope_outputs = if rope.is_some() {
        matrices[..2]
            .iter()
            .map(|m| {
                stream
                    .alloc_zeros::<f32>(m.rows() as usize)
                    .map_err(|e| failure("Buffer RoPE", e))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let mut value_output = if normalize_value {
        Some(
            stream
                .alloc_zeros::<f32>(matrices[2].rows() as usize)
                .map_err(|e| failure("Buffer V normalisé", e))?,
        )
    } else {
        None
    };
    let value_norm = if normalize_value {
        let width = head_norms.ok_or_else(|| DiagnosticError("Normes de tête requises".into()))?[1]
            .weights()
            .len();
        Some(RmsNorm::new(vec![1.0; width], norm.epsilon()).map_err(DiagnosticError)?)
    } else {
        None
    };
    let mut cache = if cache_prefix {
        let width = rope.expect("RoPE validé").width();
        let layout =
            adaforge_domain::KvLayout::new(cases as u32, matrices[1].rows() / width, width)
                .map_err(DiagnosticError)?;
        Some(super::kv_cache::KvCache::new(stream.clone(), layout)?)
    } else {
        None
    };
    let mut expected_keys = Vec::<f64>::new();
    let mut expected_values = Vec::<f64>::new();
    let mut previous_keys = Vec::<f32>::new();
    let mut previous_values = Vec::<f32>::new();
    let mut cache_report = cache_prefix.then_some(adaforge_application::CachedPrefixReport {
        tokens: 0,
        verified_values: 0,
        max_absolute_error: 0.0,
        max_tolerance_ratio: 0.0,
        previous_tokens_unchanged: true,
        overflow_rejected: false,
    });
    let mut attention_kernel = attention
        .map(|c| super::attention::AttentionKernel::new(stream.clone(), c))
        .transpose()?;
    let mut attention_report = attention.map(|c| adaforge_application::AttentionCheckReport {
        checked_values: 0,
        max_absolute_error: 0.0,
        max_tolerance_ratio: 0.0,
        window: c.window(),
        scale: c.scale(),
    });
    let mut output_kernel = tail
        .output
        .map(|b| super::attention_output::OutputKernel::new(stream.clone(), b))
        .transpose()?;
    let mut output_report = tail
        .output
        .map(|b| adaforge_application::OutputBranchReport {
            checked_values: 0,
            width: b.matrix().rows(),
            max_absolute_error: 0.0,
            max_tolerance_ratio: 0.0,
        });
    let mut ffn_kernel = tail
        .ffn
        .map(|f| super::ffn::FfnKernel::new(stream.clone(), f))
        .transpose()?;
    let mut ffn_report = tail.ffn.map(|f| adaforge_application::OutputBranchReport {
        checked_values: 0,
        width: f.width(),
        max_absolute_error: 0.0,
        max_tolerance_ratio: 0.0,
    });
    let gpu_embeddings = tail
        .embeddings
        .map(|b| super::embedding::GpuEmbeddings::new(stream.clone(), b))
        .transpose()?;
    let cpu_embeddings = tail.embeddings.map(|b| b.reference(true));
    let rms_config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut errors: Vec<ProjectionError> = matrices
        .iter()
        .map(|m| ProjectionError {
            rows: m.rows(),
            max_absolute_error: 0.0,
            max_tolerance_ratio: 0.0,
        })
        .collect();
    let mut times = Vec::new();
    let positions = if cache_prefix {
        Some((0..cases as u32).collect::<Vec<u32>>())
    } else {
        rope.map(test_positions)
    };
    for case in 0..cases {
        let position = positions.as_ref().map(|p| p[case]).unwrap_or(0);
        let mut state = 0x4090_u32 ^ if cache_prefix { case as u32 } else { 0 };
        let x: Vec<f32> = if let Some(embeddings) = cpu_embeddings.as_ref() {
            embeddings[case * columns as usize..(case + 1) * columns as usize].to_vec()
        } else {
            (0..columns)
                .map(|i| match if case == 4 { 1 } else { case % 4 } {
                    0 => 0.0,
                    1 => (i % 97) as f32 / 48.0 - 1.0,
                    2 => {
                        if i == columns / 2 {
                            1e-5
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        (state & 65535) as f32 / 32768.0 - 1.0
                    }
                })
                .collect()
        };
        // Oracle : RMSNorm FP64, arrondi FP32 à la frontière opérateur,
        // puis produit scalaire accumulé FP64. Aucun intermédiaire GPU n’est lu.
        let norm_cpu: Vec<f32> = norm
            .reference(&x)
            .map_err(DiagnosticError)?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let mut references = matrices
            .iter()
            .map(|m| m.reference_matvec(&norm_cpu).map_err(DiagnosticError))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(heads) = head_norms {
            for i in 0..2 {
                // Arrondi FP32 après projection, puis oracle RMSNorm FP64 indépendant.
                let projected: Vec<f32> = references[i].iter().map(|v| *v as f32).collect();
                references[i] = heads[i].reference(&projected).map_err(DiagnosticError)?;
            }
        }
        if let Some(rope) = rope {
            for reference in &mut references[..2] {
                let normalized: Vec<f32> = reference.iter().map(|v| *v as f32).collect();
                *reference = rope
                    .reference(&normalized, position)
                    .map_err(DiagnosticError)?;
            }
        }
        if let Some(value_norm) = value_norm.as_ref() {
            let raw_value: Vec<f32> = references[2].iter().map(|v| *v as f32).collect();
            references[2] = value_norm.reference(&raw_value).map_err(DiagnosticError)?;
        }
        if let Some(embeddings) = gpu_embeddings.as_ref() {
            embeddings.copy_row_to(case, &mut input)?;
        } else {
            stream
                .memcpy_htod(&x, &mut input)
                .map_err(|e| failure("Transfert entrée", e))?;
        }
        let executions = if !cache_prefix && case == cases - 1 {
            11
        } else {
            1
        };
        for repetition in 0..executions {
            let start = stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| failure("Début chaîne", e))?;
            {
                let mut launch = stream.launch_builder(&rms);
                launch
                    .arg(&input)
                    .arg(&scales)
                    .arg(&mut normalized)
                    .arg(&columns)
                    .arg(&epsilon);
                // SAFETY: un bloc de 256 threads ; trois buffers de columns f32,
                // sortie exclusive, dimensions validées et ressources vivantes.
                unsafe { launch.launch(rms_config) }.map_err(|e| failure("RMSNorm chaîne", e))?;
            }
            for ((matrix, weights), output) in matrices.iter().zip(&weights).zip(&mut outputs) {
                let rows = matrix.rows();
                let projection_config = LaunchConfig {
                    grid_dim: (rows.div_ceil(4), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut launch = stream.launch_builder(&project);
                launch
                    .arg(weights)
                    .arg(&normalized)
                    .arg(output)
                    .arg(&columns)
                    .arg(&rows);
                // SAFETY: stockage Q4_0 validé, entrée columns, sortie rows, bloc 128.
                // Le même stream garantit que RMSNorm a produit normalized avant lecture.
                // Aucun transfert CPU ni synchronisation hôte entre les deux kernels.
                unsafe { launch.launch(projection_config) }
                    .map_err(|e| failure("Projection chaîne", e))?;
            }
            if let Some(heads) = head_norms {
                for i in 0..2 {
                    let width = heads[i].weights().len() as u32;
                    let eps = heads[i].epsilon();
                    let config = LaunchConfig {
                        grid_dim: (matrices[i].rows() / width, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launch = stream.launch_builder(&rms);
                    launch
                        .arg(&outputs[i])
                        .arg(&head_scales[i])
                        .arg(&mut head_outputs[i])
                        .arg(&width)
                        .arg(&eps);
                    // SAFETY: 256 threads, un bloc par tête ; dimensions divisibles
                    // validées avant CUDA, buffers de rows éléments et échelles width.
                    // Même stream que la projection ; sortie distincte, pas d’alias K/V.
                    unsafe { launch.launch(config) }
                        .map_err(|e| failure("Normalisation tête", e))?;
                }
            }
            if let (Some(rope), Some(freq)) = (rope, inv_freq.as_ref()) {
                for i in 0..2 {
                    let width = rope.width();
                    let elements = matrices[i].rows();
                    let config = LaunchConfig {
                        grid_dim: ((elements / 2).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launch = stream.launch_builder(&rotary);
                    launch
                        .arg(&head_outputs[i])
                        .arg(freq)
                        .arg(&mut rope_outputs[i])
                        .arg(&width)
                        .arg(&elements)
                        .arg(&position);
                    // SAFETY: largeur paire, rows divisible par width, buffers rows
                    // et fréquences width/2 validés avant CUDA. Chaque thread écrit
                    // une paire distincte, sortie non aliasée ; même stream que RMSNorm.
                    unsafe { launch.launch(config) }.map_err(|e| failure("Rotation Q/K", e))?;
                }
            }
            if let (Some(value_norm), Some(value_output)) =
                (value_norm.as_ref(), value_output.as_mut())
            {
                let width = value_norm.weights().len() as u32;
                let config = LaunchConfig {
                    grid_dim: (matrices[2].rows() / width, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut launch = stream.launch_builder(&rms_unscaled);
                launch
                    .arg(&outputs[2])
                    .arg(value_output)
                    .arg(&width)
                    .arg(&epsilon);
                // SAFETY: bloc 256, un bloc par tête ; dimensions K/V validées,
                // entrée et sortie distinctes de rows f32. Aucun poids d’échelle
                // n’est passé ; même stream que les projections. Pas de RoPE sur V.
                unsafe { launch.launch(config) }.map_err(|e| failure("Normalisation V", e))?;
            }
            if let Some(cache) = cache.as_mut() {
                cache.append(
                    &rope_outputs[1],
                    value_output.as_ref().expect("V normalisé requis"),
                )?;
            }
            if let (Some(kernel), Some(cache)) = (attention_kernel.as_mut(), cache.as_ref()) {
                kernel.run(&rope_outputs[0], cache, position)?;
            }
            if let (Some(output), Some(attn)) = (output_kernel.as_mut(), attention_kernel.as_ref())
            {
                output.run(attn.output(), &input)?;
            }
            if let (Some(ffn), Some(output)) = (ffn_kernel.as_mut(), output_kernel.as_ref()) {
                ffn.run(output.output())?;
            }
            let end = stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| failure("Fin chaîne", e))?;
            end.synchronize()
                .map_err(|e| failure("Synchronisation chaîne", e))?;
            if cache_prefix || (case == cases - 1 && repetition > 0) {
                times.push(
                    start
                        .elapsed_ms(&end)
                        .map_err(|e| failure("Temps chaîne", e))?,
                );
            }
        }
        if let (Some(cache), Some(stats)) = (cache.as_ref(), cache_report.as_mut()) {
            expected_keys.extend_from_slice(&references[1]);
            expected_values.extend_from_slice(&references[2]);
            let (keys, values) = cache.snapshot()?;
            if !keys.starts_with(&previous_keys) || !values.starts_with(&previous_values) {
                return Err(DiagnosticError(
                    "Le cache a modifié des tokens précédents".into(),
                ));
            }
            let (ke, kr) = matvec::compare(&expected_keys, &keys)?;
            let (ve, vr) = matvec::compare(&expected_values, &values)?;
            stats.max_absolute_error = stats.max_absolute_error.max(ke).max(ve);
            stats.max_tolerance_ratio = stats.max_tolerance_ratio.max(kr).max(vr);
            stats.tokens += 1;
            stats.verified_values += keys.len() + values.len();
            if let (Some(config), Some(kernel), Some(stats)) = (
                attention,
                attention_kernel.as_ref(),
                attention_report.as_mut(),
            ) {
                let q: Vec<f32> = references[0].iter().map(|v| *v as f32).collect();
                let k: Vec<f32> = expected_keys.iter().map(|v| *v as f32).collect();
                let v: Vec<f32> = expected_values.iter().map(|v| *v as f32).collect();
                let reference = config
                    .reference(&q, &k, &v, case as u32 + 1, position)
                    .map_err(DiagnosticError)?;
                let actual = kernel.snapshot()?;
                let (error, ratio) = matvec::compare(&reference, &actual)
                    .map_err(|e| DiagnosticError(format!("Attention position {position}: {e}")))?;
                if let (Some(branch), Some(kernel), Some(out_stats)) =
                    (tail.output, output_kernel.as_ref(), output_report.as_mut())
                {
                    let attention_cpu: Vec<f32> = reference.iter().map(|v| *v as f32).collect();
                    let expected = branch
                        .reference(&attention_cpu, &x)
                        .map_err(DiagnosticError)?;
                    let actual = kernel.snapshot()?;
                    let (error, ratio) = matvec::compare(&expected, &actual).map_err(|e| {
                        DiagnosticError(format!("Branche attention position {position}: {e}"))
                    })?;
                    if let (Some(ffn), Some(kernel), Some(stats)) =
                        (tail.ffn, ffn_kernel.as_ref(), ffn_report.as_mut())
                    {
                        let branch_cpu: Vec<f32> = expected.iter().map(|v| *v as f32).collect();
                        let expected = ffn.reference(&branch_cpu).map_err(DiagnosticError)?;
                        let actual = kernel.snapshot()?;
                        let (error, ratio) = matvec::compare(&expected, &actual).map_err(|e| {
                            DiagnosticError(format!("Couche complète position {position}: {e}"))
                        })?;
                        stats.checked_values += actual.len();
                        stats.max_absolute_error = stats.max_absolute_error.max(error);
                        stats.max_tolerance_ratio = stats.max_tolerance_ratio.max(ratio);
                    }
                    out_stats.checked_values += actual.len();
                    out_stats.max_absolute_error = out_stats.max_absolute_error.max(error);
                    out_stats.max_tolerance_ratio = out_stats.max_tolerance_ratio.max(ratio);
                }
                stats.checked_values += actual.len();
                stats.max_absolute_error = stats.max_absolute_error.max(error);
                stats.max_tolerance_ratio = stats.max_tolerance_ratio.max(ratio);
            }
            previous_keys = keys;
            previous_values = values;
        }
        for (index, (reference, error_stats)) in references.iter().zip(&mut errors).enumerate() {
            let output = if index == 2 && normalize_value {
                value_output.as_ref().expect("buffer V alloué")
            } else if rope.is_some() && index < 2 {
                &rope_outputs[index]
            } else if head_norms.is_some() && index < 2 {
                &head_outputs[index]
            } else {
                &outputs[index]
            };
            let actual = stream
                .clone_dtoh(output)
                .map_err(|e| failure("Retour sortie", e))?;
            stream
                .synchronize()
                .map_err(|e| failure("Synchronisation sortie", e))?;
            let (error, ratio) = matvec::compare(reference, &actual)
                .map_err(|e| DiagnosticError(format!("Projection {index}: {e}")))?;
            error_stats.max_absolute_error = error_stats.max_absolute_error.max(error);
            error_stats.max_tolerance_ratio = error_stats.max_tolerance_ratio.max(ratio);
        }
    }
    if let (Some(cache), Some(stats)) = (cache.as_mut(), cache_report.as_mut()) {
        if cache
            .append(&rope_outputs[1], value_output.as_ref().expect("V requis"))
            .is_ok()
        {
            return Err(DiagnosticError(
                "Le cache réel a accepté un token au-delà de sa capacité".into(),
            ));
        }
        if cache.snapshot()? != (previous_keys, previous_values) {
            return Err(DiagnosticError(
                "Cache modifié après dépassement refusé".into(),
            ));
        }
        stats.overflow_rejected = true;
    }
    times.sort_by(f32::total_cmp);
    Ok(QkvReport {
        embedding_ids: tail.embeddings.map(|b| b.token_ids().to_vec()),
        ffn: ffn_report,
        output_branch: output_report,
        attention: attention_report,
        cache: cache_report,
        value_uses_key_weights: false,
        columns,
        checked_vectors: cases,
        projections: errors,
        median_kernel_ms: (times[(times.len() - 1) / 2] + times[times.len() / 2]) * 0.5,
        repetitions: times.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_dimension_mismatch_before_cuda_initialization() {
        let norm = RmsNorm::new(vec![1.0; 31], 1e-6).unwrap();
        let matrix = Q4Matrix::new(32, 1, vec![0; 18]).unwrap();
        assert!(
            check(&norm, &matrix)
                .unwrap_err()
                .to_string()
                .contains("Dimensions incompatibles")
        );
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_chain_handles_partial_rows_and_reuses_buffers() {
        let mut bytes = Vec::new();
        for i in 0..5 * 3 {
            bytes.extend(0x3800_u16.to_le_bytes());
            bytes.extend((0..16).map(|j| ((i + j) % 16) as u8 | (((i + 15 - j) % 16) as u8) << 4));
        }
        let matrix = Q4Matrix::new(96, 5, bytes).unwrap();
        let norm = RmsNorm::new((0..96).map(|i| (i % 7) as f32 - 3.0).collect(), 1e-6).unwrap();
        let report = check(&norm, &matrix).unwrap();
        assert_eq!(report.checked_vectors, 4);
        assert!(report.max_tolerance_ratio <= 1.0);
    }
}

#[cfg(test)]
mod qkv_tests {
    use super::*;
    #[test]
    fn validates_all_input_dimensions_before_cuda() {
        let norm = RmsNorm::new(vec![1.0; 32], 1e-6).unwrap();
        let a = Q4Matrix::new(32, 1, vec![0; 18]).unwrap();
        let b = Q4Matrix::new(64, 1, vec![0; 36]).unwrap();
        assert!(check_many(&norm, &[]).is_err());
        assert!(check_many(&norm, &[&a, &b, &a]).is_err());
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn distinct_qkv_weights_and_shapes() {
        fn matrix(rows: u32, seed: u8) -> Q4Matrix {
            let mut bytes = Vec::new();
            for block in 0..rows * 3 {
                bytes.extend(0x3800_u16.to_le_bytes());
                bytes.extend(
                    (0..16).map(|j| ((block as u8 + j + seed) % 16) | (((j + seed) % 16) << 4)),
                );
            }
            Q4Matrix::new(96, rows, bytes).unwrap()
        }
        let q = matrix(9, 0);
        let k = matrix(5, 1);
        let v = matrix(5, 2);
        let norm = RmsNorm::new((0..96).map(|i| (i % 7) as f32 - 3.0).collect(), 1e-6).unwrap();
        let report = check_many(&norm, &[&q, &k, &v]).unwrap();
        assert_eq!(
            report
                .projections
                .iter()
                .map(|p| p.rows)
                .collect::<Vec<_>>(),
            vec![9, 5, 5]
        );
        assert!(
            report
                .projections
                .iter()
                .all(|p| p.max_tolerance_ratio <= 1.0)
        );
    }
}

#[cfg(test)]
mod head_tests {
    use super::*;
    fn matrix(rows: u32) -> Q4Matrix {
        let mut b = Vec::new();
        for i in 0..rows * 3 {
            b.extend(0x3800_u16.to_le_bytes());
            b.extend((0..16).map(|j| ((i + j) % 16) as u8 | (((i + 15 - j) % 16) as u8) << 4));
        }
        Q4Matrix::new(96, rows, b).unwrap()
    }
    #[test]
    fn rejects_invalid_head_layout_without_gpu() {
        let q = matrix(65);
        let k = matrix(32);
        let v = matrix(32);
        let head = RmsNorm::new(vec![1.0; 32], 1e-6).unwrap();
        assert!(validate_heads(&[&q, &k, &v], &[&head, &head]).is_err());
        let q = matrix(96);
        let k = matrix(64);
        let v = matrix(64);
        assert!(validate_heads(&[&q, &k, &v], &[&head, &head]).is_err());
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_normalizes_each_head_and_preserves_raw_v() {
        for width in [32, 256, 512] {
            let q = matrix(width * 4);
            let k = matrix(width * 2);
            let norm = RmsNorm::new(vec![1.0; 96], 1e-6).unwrap();
            let qnorm =
                RmsNorm::new((0..width).map(|i| (i % 7) as f32 - 3.0).collect(), 1e-6).unwrap();
            let knorm =
                RmsNorm::new((0..width).map(|i| (i % 5) as f32 * 0.25).collect(), 1e-6).unwrap();
            let report = check_qk_norm(&norm, [&q, &k, &k], [&qnorm, &knorm]).unwrap();
            assert_eq!(report.q_heads, 4);
            assert_eq!(report.kv_heads, 2);
            assert!(
                report
                    .qkv
                    .projections
                    .iter()
                    .all(|p| p.max_tolerance_ratio <= 1.0)
            );
        }
    }
}

fn test_positions(rope: &adaforge_domain::Rope) -> Vec<u32> {
    let max = rope.context() - 1;
    [0, 0, 1, 127, 4095, max / 2, max.saturating_sub(1), max]
        .into_iter()
        .map(|p| p.min(max))
        .collect()
}
pub(crate) fn check_rope(
    norm: &RmsNorm,
    matrices: [&Q4Matrix; 3],
    heads: [&RmsNorm; 2],
    rope: &adaforge_domain::Rope,
) -> Result<adaforge_application::RopeReport, DiagnosticError> {
    check_positioned(
        norm,
        matrices,
        heads,
        rope,
        false,
        false,
        ChainTail::default(),
    )
}
pub(crate) fn check_vnorm(
    norm: &RmsNorm,
    matrices: [&Q4Matrix; 3],
    heads: [&RmsNorm; 2],
    rope: &adaforge_domain::Rope,
) -> Result<adaforge_application::RopeReport, DiagnosticError> {
    check_positioned(
        norm,
        matrices,
        heads,
        rope,
        true,
        false,
        ChainTail::default(),
    )
}
fn check_positioned(
    norm: &RmsNorm,
    matrices: [&Q4Matrix; 3],
    heads: [&RmsNorm; 2],
    rope: &adaforge_domain::Rope,
    normalize_value: bool,
    cache_prefix: bool,
    tail: ChainTail<'_>,
) -> Result<adaforge_application::RopeReport, DiagnosticError> {
    validate_heads(&matrices, &heads)?;
    if rope.width() != heads[0].weights().len() as u32 {
        return Err(DiagnosticError(
            "Largeur RoPE incompatible avec les têtes".into(),
        ));
    }
    let qkv = check_chain(
        norm,
        &matrices,
        Some(heads),
        Some(rope),
        normalize_value,
        cache_prefix,
        tail,
    )?;
    let token_count = qkv.checked_vectors as u32;
    Ok(adaforge_application::RopeReport {
        heads: adaforge_application::QkNormReport {
            qkv,
            head_width: rope.width(),
            q_heads: matrices[0].rows() / rope.width(),
            kv_heads: matrices[1].rows() / rope.width(),
        },
        positions: if cache_prefix {
            (0..token_count).collect()
        } else {
            test_positions(rope)
        },
        base: rope.base(),
        attenuated_pairs: rope.factors().iter().filter(|f| **f > 1e20).count(),
        local_attention: false,
    })
}

#[cfg(test)]
mod rope_tests {
    use super::*;
    use adaforge_domain::Rope;
    fn matrix(rows: u32) -> Q4Matrix {
        let mut b = Vec::new();
        for i in 0..rows * 3 {
            b.extend(0x3800_u16.to_le_bytes());
            b.extend((0..16).map(|j| ((i + j) % 16) as u8 | (((i + 15 - j) % 16) as u8) << 4));
        }
        Q4Matrix::new(96, rows, b).unwrap()
    }
    #[test]
    fn incompatible_rope_width_fails_before_cuda() {
        let q = matrix(64);
        let k = matrix(32);
        let norm = RmsNorm::new(vec![1.0; 96], 1e-6).unwrap();
        let head = RmsNorm::new(vec![1.0; 32], 1e-6).unwrap();
        let rope = Rope::new(64, 10000.0, vec![1.0; 32], 100).unwrap();
        assert!(check_rope(&norm, [&q, &k, &k], [&head, &head], &rope).is_err());
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_rotates_local_and_proportional_heads_at_long_positions() {
        for (width, base, proportional) in [
            (32, 10000.0, false),
            (256, 10000.0, false),
            (512, 1000000.0, true),
        ] {
            let q = matrix(width * 4);
            let k = matrix(width * 2);
            let norm = RmsNorm::new(vec![1.0; 96], 1e-6).unwrap();
            let head =
                RmsNorm::new((0..width).map(|i| (i % 7) as f32 - 3.0).collect(), 1e-6).unwrap();
            let factors = (0..width / 2)
                .map(|i| {
                    if proportional && i >= width / 8 {
                        1e30
                    } else {
                        1.0
                    }
                })
                .collect();
            let rope = Rope::new(width, base, factors, 262144).unwrap();
            let report = check_rope(&norm, [&q, &k, &k], [&head, &head], &rope).unwrap();
            assert_eq!(report.heads.qkv.checked_vectors, 8);
            assert_eq!(*report.positions.last().unwrap(), 262143);
            assert!(
                report
                    .heads
                    .qkv
                    .projections
                    .iter()
                    .all(|p| p.max_tolerance_ratio <= 1.0)
            );
        }
    }
}

#[cfg(test)]
mod value_tests {
    use super::*;
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn value_uses_raw_projection_without_key_scale_or_rope() {
        for width in [32, 256, 512] {
            let mut bytes = Vec::new();
            for i in 0..width * 2 * 3 {
                bytes.extend(0x3800_u16.to_le_bytes());
                bytes.extend(
                    (0..16).map(|j| ((i + j) % 16) as u8 | (((i + 15 - j) % 16) as u8) << 4),
                );
            }
            let matrix = Q4Matrix::new(96, width * 2, bytes).unwrap();
            let norm = RmsNorm::new(vec![1.0; 96], 1e-6).unwrap();
            let head =
                RmsNorm::new((0..width).map(|i| (i % 7) as f32 - 3.0).collect(), 1e-6).unwrap();
            let rope =
                adaforge_domain::Rope::new(width, 10000.0, vec![1.0; width as usize / 2], 262144)
                    .unwrap();
            let report =
                check_vnorm(&norm, [&matrix, &matrix, &matrix], [&head, &head], &rope).unwrap();
            assert!(
                report
                    .heads
                    .qkv
                    .projections
                    .iter()
                    .all(|p| p.max_tolerance_ratio <= 1.0)
            );
        }
    }
}

pub(crate) fn check_cached_kv(
    norm: &RmsNorm,
    matrices: [&Q4Matrix; 3],
    heads: [&RmsNorm; 2],
    rope: &adaforge_domain::Rope,
) -> Result<adaforge_application::RopeReport, DiagnosticError> {
    check_positioned(
        norm,
        matrices,
        heads,
        rope,
        true,
        true,
        ChainTail::default(),
    )
}
#[cfg(test)]
mod cached_tests {
    use super::*;
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn projection_outputs_are_cached_in_order_without_host_round_trip() {
        let width = 32;
        let rows = 64;
        let mut bytes = Vec::new();
        for i in 0..rows * 3 {
            bytes.extend(0x3800_u16.to_le_bytes());
            bytes.extend((0..16).map(|j| ((i + j) % 16) as u8 | (((i + 15 - j) % 16) as u8) << 4));
        }
        let matrix = Q4Matrix::new(96, rows, bytes).unwrap();
        let norm = RmsNorm::new(vec![1.0; 96], 1e-6).unwrap();
        let head = RmsNorm::new((0..width).map(|i| (i % 7) as f32 - 3.0).collect(), 1e-6).unwrap();
        let rope = adaforge_domain::Rope::new(width, 10000.0, vec![1.0; 16], 100).unwrap();
        let report =
            check_cached_kv(&norm, [&matrix, &matrix, &matrix], [&head, &head], &rope).unwrap();
        let stats = report.heads.qkv.cache.unwrap();
        assert_eq!(stats.tokens, 8);
        assert!(stats.overflow_rejected && stats.previous_tokens_unchanged);
        assert_eq!(stats.verified_values, 36 * 64 * 2);
    }
}

pub(crate) fn check_attention(
    norm: &RmsNorm,
    matrices: [&Q4Matrix; 3],
    heads: [&RmsNorm; 2],
    rope: &adaforge_domain::Rope,
    config: adaforge_domain::AttentionConfig,
) -> Result<adaforge_application::RopeReport, DiagnosticError> {
    if config.width() != rope.width() {
        return Err(DiagnosticError(
            "Largeur attention incompatible avec RoPE".into(),
        ));
    }
    check_positioned(
        norm,
        matrices,
        heads,
        rope,
        true,
        true,
        ChainTail {
            attention: Some(config),
            output: None,
            ffn: None,
            embeddings: None,
        },
    )
}

pub(crate) fn check_attention_branch(
    norm: &RmsNorm,
    matrices: [&Q4Matrix; 3],
    heads: [&RmsNorm; 2],
    rope: &adaforge_domain::Rope,
    config: adaforge_domain::AttentionConfig,
    branch: &adaforge_domain::AttentionOutput,
) -> Result<adaforge_application::RopeReport, DiagnosticError> {
    if config.width() != rope.width() {
        return Err(DiagnosticError(
            "Largeur attention/RoPE incompatible".into(),
        ));
    }
    check_positioned(
        norm,
        matrices,
        heads,
        rope,
        true,
        true,
        ChainTail {
            attention: Some(config),
            output: Some(branch),
            ffn: None,
            embeddings: None,
        },
    )
}

pub(crate) fn check_layer(
    weights: adaforge_application::LayerWeights<'_>,
) -> Result<adaforge_application::RopeReport, DiagnosticError> {
    if weights.attention.width() != weights.rope.width() {
        return Err(DiagnosticError(
            "Largeur attention/RoPE incompatible".into(),
        ));
    }
    check_positioned(
        weights.input_norm,
        weights.matrices,
        weights.head_norms,
        weights.rope,
        true,
        true,
        ChainTail {
            attention: Some(weights.attention),
            output: Some(weights.output),
            ffn: Some(weights.ffn),
            embeddings: None,
        },
    )
}

#[cfg(test)]
mod layer_tests {
    use super::*;
    use adaforge_domain::{AttentionConfig, AttentionOutput, Ffn, Rope};
    fn matrix(columns: u32, rows: u32, seed: u32) -> Q4Matrix {
        let mut b = Vec::new();
        for i in 0..columns / 32 * rows {
            b.extend(0x2800_u16.to_le_bytes());
            b.extend(
                (0..16)
                    .map(|j| ((i + j + seed) % 16) as u8 | (((i + 15 - j + seed) % 16) as u8) << 4),
            );
        }
        Q4Matrix::new(columns, rows, b).unwrap()
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn full_dense_layer_matches_independent_reference() {
        let input_norm = RmsNorm::new(vec![1.0; 32], 1e-6).unwrap();
        let q = matrix(32, 64, 0);
        let k = matrix(32, 32, 1);
        let v = matrix(32, 32, 2);
        let qnorm = RmsNorm::new(vec![0.75; 32], 1e-6).unwrap();
        let knorm = RmsNorm::new(vec![1.25; 32], 1e-6).unwrap();
        let output = AttentionOutput::new(
            matrix(64, 32, 3),
            RmsNorm::new(vec![1.0; 32], 1e-6).unwrap(),
        )
        .unwrap();
        let ffn = Ffn::new(
            RmsNorm::new(vec![1.0; 32], 1e-6).unwrap(),
            matrix(32, 64, 4),
            matrix(32, 64, 5),
            matrix(64, 32, 6),
            RmsNorm::new(vec![0.75; 32], 1e-6).unwrap(),
            0.5,
        )
        .unwrap();
        let rope = Rope::new(32, 10000.0, vec![1.0; 16], 100).unwrap();
        let report = check_layer(adaforge_application::LayerWeights {
            input_norm: &input_norm,
            matrices: [&q, &k, &v],
            head_norms: [&qnorm, &knorm],
            rope: &rope,
            attention: AttentionConfig::new(2, 1, 32, Some(3), 1.0).unwrap(),
            output: &output,
            ffn: &ffn,
        })
        .unwrap();
        let stats = report.heads.qkv.ffn.unwrap();
        assert_eq!(stats.checked_values, 8 * 32);
        assert!(stats.max_tolerance_ratio <= 1.0);
    }
}

pub(crate) fn check_embedded_layer(
    weights: adaforge_application::LayerWeights<'_>,
    embeddings: &adaforge_domain::EmbeddingBatch,
) -> Result<adaforge_application::RopeReport, DiagnosticError> {
    if weights.attention.width() != weights.rope.width() {
        return Err(DiagnosticError(
            "Largeur attention/RoPE incompatible".into(),
        ));
    }
    check_positioned(
        weights.input_norm,
        weights.matrices,
        weights.head_norms,
        weights.rope,
        true,
        true,
        ChainTail {
            attention: Some(weights.attention),
            output: Some(weights.output),
            ffn: Some(weights.ffn),
            embeddings: Some(embeddings),
        },
    )
}
