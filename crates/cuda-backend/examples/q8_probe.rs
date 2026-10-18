//! Experimental kernel probe; does not change production inference.
use adaforge_application::ModelInspector;
use adaforge_model_formats::GgufInspector;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{error::Error, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let model = std::env::args().nth(1).ok_or("model path required")?;
    let inspector = GgufInspector::default();
    let ctx = CudaContext::new(0)?;
    if !ctx.name()?.contains("RTX 4090") {
        return Err("RTX 4090 required".into());
    }
    let stream = ctx.default_stream();
    let source = format!(
        "{}\n{}",
        include_str!("../../../kernels/ada/q4_matvec.cu"),
        include_str!("../../../kernels/ada/q4_q8.cu")
    );
    let module = ctx.load_module(compile_ptx_with_opts(
        source,
        CompileOptions {
            arch: Some("compute_89"),
            ..Default::default()
        },
    )?)?;
    let quant = module.load_function("quantize_q8")?;
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
    ] {
        let matrix = inspector.read_q4_matrix(Path::new(&model), name)?;
        let columns = matrix.columns();
        let rows = matrix.rows();
        let blocks = columns as usize / 32;
        let weights = stream.clone_htod(matrix.bytes())?;
        let mut q = stream.alloc_zeros::<i8>(columns as usize)?;
        let mut scales = stream.alloc_zeros::<f32>(blocks)?;
        let mut output = stream.alloc_zeros::<f32>(rows as usize)?;
        let mut state = 0x4090u32;
        for case in 0..4 {
            let input: Vec<f32> = (0..columns)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    match case {
                        0 => 0.0,
                        1 => (i % 97) as f32 / 48.0 - 1.0,
                        2 => ((state & 65535) as f32 / 32768.0 - 1.0) * 0.125,
                        _ => {
                            if i % 32 == 0 {
                                10.0
                            } else {
                                0.001
                            }
                        }
                    }
                })
                .collect();
            let device = stream.clone_htod(&input)?;
            let qconfig = LaunchConfig {
                grid_dim: (columns.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = stream.launch_builder(&quant);
            launch
                .arg(&device)
                .arg(&mut q)
                .arg(&mut scales)
                .arg(&columns);
            // SAFETY: columns is a positive multiple of 32; each warp owns one full block.
            unsafe { launch.launch(qconfig) }?;
            let hostq = stream.clone_dtoh(&q)?;
            let hostscales = stream.clone_dtoh(&scales)?;
            stream.synchronize()?;
            let mut reconstructed = vec![0f32; input.len()];
            for b in 0..blocks {
                let maximum = input[b * 32..b * 32 + 32]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0f32, f32::max);
                let d = maximum / 127.0;
                assert_eq!(hostscales[b].to_bits(), d.to_bits());
                for j in 0..32 {
                    let i = b * 32 + j;
                    let expected = if d == 0.0 {
                        0
                    } else {
                        (input[i] / d).round_ties_even().clamp(-127.0, 127.0) as i8
                    };
                    assert_eq!(hostq[i], expected);
                    reconstructed[i] = d * f32::from(expected);
                }
            }
            let original = matrix.reference_matvec(&input)?;
            let oracle = matrix.reference_matvec(&reconstructed)?;
            for (kernel, kpar) in [
                ("q4_q8_matvec", false),
                ("q4_q8_kparallel", true),
                (
                    if columns == 15360 {
                        "q4_matvec_kparallel"
                    } else {
                        "q4_matvec"
                    },
                    columns == 15360,
                ),
            ] {
                let func = module.load_function(kernel)?;
                let isq8 = kernel.starts_with("q4_q8");
                let config = LaunchConfig {
                    grid_dim: (if kpar { rows } else { rows.div_ceil(4) }, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut times = Vec::new();
                for rep in 0..21 {
                    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    if isq8 {
                        let mut l = stream.launch_builder(&quant);
                        l.arg(&device).arg(&mut q).arg(&mut scales).arg(&columns);
                        // SAFETY: same validated quant buffers and dimensions as above.
                        unsafe { l.launch(qconfig) }?;
                    }
                    let mut l = stream.launch_builder(&func);
                    l.arg(&weights);
                    if isq8 {
                        l.arg(&q).arg(&scales);
                    } else {
                        l.arg(&device);
                    }
                    l.arg(&mut output).arg(&columns).arg(&rows);
                    // SAFETY: validated weights, input/output dimensions, full 128-thread blocks.
                    unsafe { l.launch(config) }?;
                    let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    end.synchronize()?;
                    if rep > 0 {
                        times.push(start.elapsed_ms(&end)?);
                    }
                }
                let actual = stream.clone_dtoh(&output)?;
                stream.synchronize()?;
                let expected = if isq8 { &oracle } else { &original };
                let mut max_error = 0f64;
                let mut noise = 0f64;
                let mut signal = 0f64;
                for ((&y, &refq), &reforiginal) in actual.iter().zip(expected).zip(&original) {
                    assert!(y.is_finite());
                    let err = (f64::from(y) - refq).abs();
                    assert!(
                        err <= 1e-4 + 1e-4 * refq.abs(),
                        "kernel mismatch {kernel}: {err}"
                    );
                    let change = (f64::from(y) - reforiginal).abs();
                    max_error = max_error.max(change);
                    noise += change * change;
                    signal += reforiginal * reforiginal;
                }
                times.sort_by(f32::total_cmp);
                println!(
                    "Q8_PROBE {{\"tensor\":\"{name}\",\"case\":{case},\"kernel\":\"{kernel}\",\"quant_included\":{isq8},\"median_ms\":{},\"max_absolute_error_vs_fp32\":{max_error},\"relative_l2_vs_fp32\":{}}}",
                    times[times.len() / 2],
                    if signal > 0.0 {
                        (noise / signal).sqrt()
                    } else {
                        0.0
                    }
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn q8_rounds_ties_even_and_propagates_nonfinite_blocks() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let source = format!(
            "{}\n{}",
            include_str!("../../../kernels/ada/q4_matvec.cu"),
            include_str!("../../../kernels/ada/q4_q8.cu")
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
        let quant = module.load_function("quantize_q8").unwrap();
        let columns = 160u32;
        let mut host = vec![0f32; columns as usize];
        host[..6].copy_from_slice(&[127.0, 0.5, 1.5, -0.5, -1.5, 2.5]);
        host[64] = f32::NAN;
        host[96] = f32::INFINITY;
        host[128] = f32::NEG_INFINITY;
        let input = stream.clone_htod(&host).unwrap();
        let mut q = stream.alloc_zeros::<i8>(columns as usize).unwrap();
        let mut scales = stream.alloc_zeros::<f32>(5).unwrap();
        let mut launch = stream.launch_builder(&quant);
        launch
            .arg(&input)
            .arg(&mut q)
            .arg(&mut scales)
            .arg(&columns);
        // SAFETY: two full CTAs cover five complete 32-element groups; extra warps return.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (2, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
        let values = stream.clone_dtoh(&q).unwrap();
        let scales = stream.clone_dtoh(&scales).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(&values[..6], &[127, 0, 2, 0, -2, 2]);
        assert_eq!(scales[0], 1.0);
        assert_eq!(scales[1], 0.0);
        assert!(values[32..].iter().all(|&q| q == 0));
        assert!(scales[2..].iter().all(|s| s.is_nan()));
    }
}
