//! Combined FFN gate/up: cache hints versus production tiled reduction, rotating timed order.
use adaforge_application::ModelInspector;
use adaforge_model_formats::GgufInspector;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{error::Error, path::Path};
fn main() -> Result<(), Box<dyn Error>> {
    let model = std::env::args().nth(1).ok_or("model path required")?;
    let ctx = CudaContext::new(0)?;
    if !ctx.name()?.contains("RTX 4090") {
        return Err("RTX 4090 required".into());
    }
    let stream = ctx.default_stream();
    let source = format!(
        "{}\n{}\n{}",
        include_str!("../../../kernels/ada/q4_matvec.cu"),
        include_str!("../../../kernels/ada/q4_tiled_probe.cu"),
        include_str!("../../../kernels/ada/q4_cache_probe.cu")
    );
    let module = ctx.load_module(compile_ptx_with_opts(
        source,
        CompileOptions {
            arch: Some("compute_89"),
            ..Default::default()
        },
    )?)?;
    let tiled = module.load_function("q4_cache_cg")?;
    let split = module.load_function("q4_cache_cs")?;
    let asynchronous = module.load_function("q4_cache_ca")?;
    let asynchronous4 = module.load_function("q4_cache_nc")?;
    let evict = module.load_function("probe_evict")?;
    let count = 32 * 1024 * 1024u32;
    let mut trash = stream.alloc_zeros::<u32>(count as usize)?;
    let inspector = GgufInspector::default();
    let mut matrices = Vec::new();
    for layer in [0, 5] {
        let gate =
            inspector.read_q4_matrix(Path::new(&model), &format!("blk.{layer}.ffn_gate.weight"))?;
        let up =
            inspector.read_q4_matrix(Path::new(&model), &format!("blk.{layer}.ffn_up.weight"))?;
        let mut bytes = gate.bytes().to_vec();
        bytes.extend_from_slice(up.bytes());
        matrices.push((
            format!("blk.{layer}.gate_up"),
            adaforge_domain::Q4Matrix::new(gate.columns(), gate.rows() + up.rows(), bytes)?,
        ));
    }
    for rows in [1u32, 5, 9] {
        let columns = 96;
        let mut bytes = Vec::new();
        for block in 0..rows * columns / 32 {
            bytes.extend([0x3800u16, 0xbc00, 1, 0][block as usize % 4].to_le_bytes());
            bytes.extend((0..16).map(|i| (i * 37 + block * 11) as u8));
        }
        matrices.push((
            format!("synthetic-{rows}x{columns}"),
            adaforge_domain::Q4Matrix::new(columns, rows, bytes)?,
        ));
    }
    for (name, m) in matrices {
        let rows = m.rows();
        let columns = m.columns();
        let blocks = columns as usize / 32;
        let tiles = (rows as usize).div_ceil(4) * blocks;
        let mut codes = vec![0u16; tiles * 32];
        let mut scales = vec![0u16; tiles * 4];
        for row in 0..rows as usize {
            for b in 0..blocks {
                let p = (row * blocks + b) * 18;
                let tile = (row / 4) * blocks + b;
                let r = row % 4;
                scales[tile * 4 + r] = u16::from_le_bytes(m.bytes()[p..p + 2].try_into()?);
                for word in 0..8 {
                    codes[(tile * 8 + word) * 4 + r] = u16::from_le_bytes(
                        m.bytes()[p + 2 + word * 2..p + 4 + word * 2].try_into()?,
                    );
                }
            }
        }
        // Exact inverse verifies every scale and nibble, including signed scales.
        let mut inverse = vec![0u8; m.bytes().len()];
        for row in 0..rows as usize {
            for b in 0..blocks {
                let p = (row * blocks + b) * 18;
                let tile = (row / 4) * blocks + b;
                let r = row % 4;
                inverse[p..p + 2].copy_from_slice(&scales[tile * 4 + r].to_le_bytes());
                for word in 0..8 {
                    inverse[p + 2 + word * 2..p + 4 + word * 2]
                        .copy_from_slice(&codes[(tile * 8 + word) * 4 + r].to_le_bytes());
                }
            }
        }
        assert_eq!(inverse, m.bytes());

        let codes = stream.clone_htod(&codes)?;
        let scales = stream.clone_htod(&scales)?;
        let old = module.load_function("q4_tiled_split")?;
        let mut out = stream.alloc_zeros::<f32>(rows as usize)?;
        let mut state = 0x4090u32;
        for case in 0..3 {
            let x: Vec<f32> = (0..columns)
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
            let expected = m.reference_matvec(&x)?;
            let x = stream.clone_htod(&x)?;
            for cold in [false, true] {
                let mut native_bits = Vec::new();
                let mut times: [Vec<f32>; 5] = std::array::from_fn(|_| Vec::new());
                for rep in 0..22 {
                    for order in 0..5 {
                        let variant = (order + rep) % 5;
                        if cold {
                            let mut l = stream.launch_builder(&evict);
                            l.arg(&mut trash).arg(&count);
                            // SAFETY: one bounded write per element of the allocated eviction buffer.
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
                        if variant == 0 {
                            let mut l = stream.launch_builder(&old);
                            l.arg(&codes)
                                .arg(&scales)
                                .arg(&x)
                                .arg(&mut out)
                                .arg(&columns)
                                .arg(&rows);
                            // SAFETY: native validated matrix and bounded rows; down uses four warps per row.
                            unsafe {
                                l.launch(LaunchConfig {
                                    grid_dim: (rows.div_ceil(4), 1, 1),
                                    block_dim: (128, 1, 1),
                                    shared_mem_bytes: 0,
                                })
                            }?;
                        } else {
                            let mut l = stream.launch_builder(match variant {
                                1 => &tiled,
                                2 => &split,
                                3 => &asynchronous,
                                _ => &asynchronous4,
                            });
                            l.arg(&codes)
                                .arg(&scales)
                                .arg(&x)
                                .arg(&mut out)
                                .arg(&columns)
                                .arg(&rows);
                            // SAFETY: padded four-row tiles; 128 threads per CTA and bounded row stores.
                            unsafe {
                                l.launch(LaunchConfig {
                                    grid_dim: (rows.div_ceil(4), 1, 1),
                                    block_dim: (128, 1, 1),
                                    shared_mem_bytes: 0,
                                })
                            }?;
                        }
                        let end =
                            stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                        end.synchronize()?;
                        if rep >= 2 {
                            times[variant].push(start.elapsed_ms(&end)?);
                        }
                        if rep == 0 {
                            let actual = stream.clone_dtoh(&out)?;
                            stream.synchronize()?;
                            if variant == 0 {
                                native_bits =
                                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                            } else if variant >= 1 {
                                assert_eq!(
                                    native_bits,
                                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    "native reduction tree changed"
                                );
                            }
                            for (&a, &e) in actual.iter().zip(&expected) {
                                assert!(
                                    a.is_finite()
                                        && (f64::from(a) - e).abs() <= 1e-4 + 1e-4 * e.abs(),
                                    "oracle mismatch"
                                );
                            }
                        }
                    }
                }
                for t in &mut times {
                    t.sort_by(f32::total_cmp);
                }
                println!(
                    "CACHE_PROBE tensor={name} case={case} cold={cold} native_ms={} cg_ms={} cs_ms={} ca_ms={} nc_ms={} lossless=true oracle=true",
                    (times[0][9] + times[0][10]) * 0.5,
                    (times[1][9] + times[1][10]) * 0.5,
                    (times[2][9] + times[2][10]) * 0.5,
                    (times[3][9] + times[3][10]) * 0.5,
                    (times[4][9] + times[4][10]) * 0.5
                );
            }
        }
    }
    Ok(())
}
