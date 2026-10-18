//! Native Q4 schedules on real local/global shapes; no inference changes.
use adaforge_application::ModelInspector;
use adaforge_model_formats::GgufInspector;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{error::Error, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let model = std::env::args().nth(1).ok_or("model required")?;
    let ctx = CudaContext::new(0)?;
    if !ctx.name()?.contains("RTX 4090") {
        return Err("RTX 4090 required".into());
    }
    let stream = ctx.default_stream();
    let source = format!(
        "{}\n{}\n{}",
        include_str!("../../../kernels/ada/q4_matvec.cu"),
        include_str!("../../../kernels/ada/q4_shape_probe.cu"),
        include_str!("../../../kernels/ada/q4_tiled_probe.cu")
    );
    let module = ctx.load_module(compile_ptx_with_opts(
        source,
        CompileOptions {
            arch: Some("compute_89"),
            ..Default::default()
        },
    )?)?;
    let names = [
        "q4_matvec",
        "q4_matvec_kparallel",
        "q4_shape1",
        "q4_shape2",
        "q4_shape4",
    ];
    let kernels = names
        .iter()
        .map(|n| module.load_function(n))
        .collect::<Result<Vec<_>, _>>()?;
    let evict = module.load_function("probe_evict")?;
    let count = 32 * 1024 * 1024u32;
    let mut trash = stream.alloc_zeros::<u32>(count as usize)?;
    let inspector = GgufInspector::default();
    let mut matrices = Vec::new();
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.5.attn_q.weight",
        "blk.5.attn_k.weight",
        "blk.0.attn_output.weight",
        "blk.5.attn_output.weight",
        "blk.0.ffn_down.weight",
    ] {
        matrices.push((
            name.to_owned(),
            inspector.read_q4_matrix(Path::new(&model), name)?,
        ));
    }
    for rows in [1u32, 5, 9] {
        let mut bytes = Vec::new();
        for b in 0..rows * 3 {
            bytes.extend([0u16, 1, 0x3800, 0xbc00][b as usize % 4].to_le_bytes());
            bytes.extend((0..16).map(|i| (i * 37 + b * 11) as u8));
        }
        matrices.push((
            format!("synthetic-{rows}x96"),
            adaforge_domain::Q4Matrix::new(96, rows, bytes)?,
        ));
    }
    for (name, m) in matrices {
        let rows = m.rows();
        let columns = m.columns();
        let weights = stream.clone_htod(m.bytes())?;
        let mut output = stream.alloc_zeros::<f32>(rows as usize)?;
        let mut state = 0x4090u32;
        for case in 0..3 {
            let input: Vec<f32> = (0..columns)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    match case {
                        0 => 0.,
                        1 => (i % 97) as f32 / 48. - 1.,
                        _ => (state & 65535) as f32 / 32768. - 1.,
                    }
                })
                .collect();
            let oracle = m.reference_matvec(&input)?;
            let input = stream.clone_htod(&input)?;
            for cold in [false, true] {
                let mut times: [Vec<f32>; 5] = std::array::from_fn(|_| Vec::new());
                let mut reference_bits = Vec::new();
                for rep in 0..32 {
                    for order in 0..5 {
                        let variant = (order + rep) % 5;
                        if cold {
                            let mut l = stream.launch_builder(&evict);
                            l.arg(&mut trash).arg(&count);
                            // SAFETY: bounded writes into the allocated 128 MiB buffer.
                            unsafe {
                                l.launch(LaunchConfig {
                                    grid_dim: (count.div_ceil(256), 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                })
                            }?;
                        }
                        let start =
                            stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                        let per_cta = [4, 1, 1, 2, 4][variant];
                        let mut l = stream.launch_builder(&kernels[variant]);
                        l.arg(&weights)
                            .arg(&input)
                            .arg(&mut output)
                            .arg(&columns)
                            .arg(&rows);
                        // SAFETY: native validated matrix, bounded stores and 128 threads;
                        // all warps in the research schedules reach the shared reduction.
                        unsafe {
                            l.launch(LaunchConfig {
                                grid_dim: (rows.div_ceil(per_cta), 1, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            })
                        }?;
                        let end =
                            stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                        end.synchronize()?;
                        if rep >= 2 {
                            times[variant].push(start.elapsed_ms(&end)?);
                        }
                        if rep == 0 {
                            let actual = stream.clone_dtoh(&output)?;
                            stream.synchronize()?;
                            for (&a, &e) in actual.iter().zip(&oracle) {
                                assert!(
                                    a.is_finite()
                                        && (f64::from(a) - e).abs() <= 1e-4 + 1e-4 * e.abs(),
                                    "oracle mismatch {name} {variant}"
                                );
                            }
                            let bits = actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                            if variant == 0 {
                                reference_bits = bits;
                            } else if variant >= 2 {
                                assert_eq!(
                                    bits, reference_bits,
                                    "FP32 tree changed {name} {variant}"
                                );
                            }
                        }
                    }
                }
                for (variant, t) in times.iter_mut().enumerate() {
                    t.sort_by(f32::total_cmp);
                    println!(
                        "SHAPE_PROBE tensor={name} case={case} cold={cold} kernel={} ms={} cpu_oracle=true native_bits={}",
                        names[variant],
                        (t[14] + t[15]) * 0.5,
                        variant != 1
                    );
                }
            }
        }
    }
    Ok(())
}
