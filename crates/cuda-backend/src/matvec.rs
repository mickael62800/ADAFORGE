use super::failure;
use adaforge_application::{DiagnosticError, MatvecReport};
use adaforge_domain::Q4Matrix;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

pub(crate) fn check(matrix: &Q4Matrix) -> Result<MatvecReport, DiagnosticError> {
    let ctx = CudaContext::new(0).map_err(|e| failure("Initialisation CUDA", e))?;
    let name = ctx.name().map_err(|e| failure("Nom GPU", e))?;
    let major = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| failure("CUDA", e))?;
    let minor = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| failure("CUDA", e))?;
    if (major, minor) != (8, 9) || !name.contains("RTX 4090") {
        return Err(DiagnosticError("RTX 4090 sm_89 requise".into()));
    }
    let ptx = compile_ptx_with_opts(
        include_str!("../../../kernels/ada/q4_matvec.cu"),
        CompileOptions {
            arch: Some("compute_89"),
            ..Default::default()
        },
    )
    .map_err(|e| failure("Compilation Q4_0", e))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| failure("Module Q4_0", e))?;
    let function = module
        .load_function("q4_matvec")
        .map_err(|e| failure("Kernel Q4_0", e))?;
    let stream = ctx.default_stream();
    let weights = stream
        .clone_htod(matrix.bytes())
        .map_err(|e| failure("Transfert poids", e))?;
    let rows = matrix.rows();
    let columns = matrix.columns();
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(4), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut output = stream
        .alloc_zeros::<f32>(rows as usize)
        .map_err(|e| failure("Allocation sortie", e))?;
    let mut max_absolute_error = 0.0_f64;
    let mut max_tolerance_ratio = 0.0_f64;
    let mut times = Vec::new();
    // Zéro, signal structuré et pseudo-aléatoire reproductible, amplitudes différentes.
    for case in 0..3 {
        let mut state = 0x4090_u32;
        let x: Vec<f32> = (0..columns)
            .map(|i| match case {
                0 => 0.0,
                1 => (i % 97) as f32 / 48.0 - 1.0,
                _ => {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    ((state & 65535) as f32 / 32768.0 - 1.0) * 0.125
                }
            })
            .collect();
        let reference = matrix.reference_matvec(&x).map_err(DiagnosticError)?;
        let input = stream
            .clone_htod(&x)
            .map_err(|e| failure("Transfert vecteur", e))?;
        // Un échauffement et dix mesures, sur le dernier vecteur seulement.
        let executions = if case == 2 { 11 } else { 1 };
        for repetition in 0..executions {
            let start = stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| failure("Événement début", e))?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&weights)
                .arg(&input)
                .arg(&mut output)
                .arg(&columns)
                .arg(&rows);
            // SAFETY: Q4Matrix valide dimensions et stockage (18 octets / 32 poids),
            // échelles finies, x a columns éléments, y rows. Bloc exactement 128 threads.
            // Les lignes hors limites quittent par warp entier ; les ressources vivent
            // jusqu’à la synchronisation et la sortie est empruntée exclusivement.
            unsafe { launch.launch(config) }.map_err(|e| failure("Lancement matvec", e))?;
            let end = stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| failure("Événement fin", e))?;
            end.synchronize()
                .map_err(|e| failure("Synchronisation matvec", e))?;
            if case == 2 && repetition > 0 {
                times.push(
                    start
                        .elapsed_ms(&end)
                        .map_err(|e| failure("Chronométrage", e))?,
                );
            }
        }
        let actual = stream
            .clone_dtoh(&output)
            .map_err(|e| failure("Retour résultat", e))?;
        stream
            .synchronize()
            .map_err(|e| failure("Synchronisation retour", e))?;
        let (error, ratio) = compare(&reference, &actual)?;
        max_absolute_error = max_absolute_error.max(error);
        max_tolerance_ratio = max_tolerance_ratio.max(ratio);
    }
    times.sort_by(f32::total_cmp);
    Ok(MatvecReport {
        rows,
        columns,
        checked_vectors: 3,
        max_absolute_error,
        max_tolerance_ratio,
        median_kernel_ms: (times[4] + times[5]) * 0.5,
        repetitions: times.len(),
    })
}

/// Thread-major repack of a row-major Q4_0 matrix for `q4_tm_row`.
/// `partitions` is the number of warps sharing one row (1 or 4).
/// Returns (codes, scales); byte layout documented in q4_thread_major.cu.
pub(crate) fn thread_major(matrix: &Q4Matrix, partitions: usize) -> (Vec<u8>, Vec<u8>) {
    let bytes = matrix.bytes();
    let rows = matrix.rows() as usize;
    let blocks = matrix.columns() as usize / 32;
    let stride = 4 * partitions;
    let iterations = blocks.div_ceil(stride);
    let (chunks, tail) = (iterations / 8, iterations % 8);
    let code_bytes = chunks * 512 + tail * 64;
    let scale_bytes = chunks * 64 + tail * 8;
    let slots = rows * partitions;
    let mut codes = vec![0u8; slots * code_bytes];
    let mut scales = vec![0u8; slots * scale_bytes];
    for row in 0..rows {
        for part in 0..partitions {
            let slot = row * partitions + part;
            let (cbase, sbase) = (slot * code_bytes, slot * scale_bytes);
            for t in 0..iterations {
                let (c, i) = (t / 8, t % 8);
                for group in 0..4 {
                    let b = part * 4 + group + stride * t;
                    if b >= blocks {
                        continue;
                    }
                    let p = (row * blocks + b) * 18;
                    let s = if c < chunks {
                        sbase + (c * 4 + group) * 16 + i * 2
                    } else {
                        sbase + chunks * 64 + (i * 4 + group) * 2
                    };
                    scales[s..s + 2].copy_from_slice(&bytes[p..p + 2]);
                    for word in 0..8 {
                        let lane = group * 8 + word;
                        let o = if c < chunks {
                            cbase + (c * 32 + lane) * 16 + i * 2
                        } else {
                            cbase + chunks * 512 + (i * 32 + lane) * 2
                        };
                        let q = p + 2 + word * 2;
                        codes[o..o + 2].copy_from_slice(&bytes[q..q + 2]);
                    }
                }
            }
        }
    }
    (codes, scales)
}

pub(super) fn compare(reference: &[f64], actual: &[f32]) -> Result<(f64, f64), DiagnosticError> {
    if reference.len() != actual.len() {
        return Err(DiagnosticError("Tailles de résultats incompatibles".into()));
    }
    let mut max_error = 0.0_f64;
    let mut max_ratio = 0.0_f64;
    for (i, (&expected, &value)) in reference.iter().zip(actual).enumerate() {
        let error = (f64::from(value) - expected).abs();
        let ratio = error / (1e-4 + 1e-4 * expected.abs());
        if !expected.is_finite() || !value.is_finite() || ratio > 1.0 {
            return Err(DiagnosticError(format!(
                "Matvec incorrect ligne {i}: CPU {expected}, GPU {value}, erreur {error}"
            )));
        }
        max_error = max_error.max(error);
        max_ratio = max_ratio.max(ratio);
    }
    Ok((max_error, max_ratio))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_rejects_nonfinite_and_wrong_values() {
        assert!(compare(&[1.0], &[1.0]).is_ok());
        for v in [f32::NAN, f32::INFINITY, 2.0] {
            assert!(compare(&[1.0], &[v]).is_err());
        }
        assert!(compare(&[1.0], &[]).is_err());
    }
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn kparallel_qkv_routes_partial_rows_and_preserves_unused_v() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(
                compile_ptx_with_opts(
                    include_str!("../../../kernels/ada/q4_matvec.cu"),
                    CompileOptions {
                        arch: Some("compute_89"),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let kernel = module.load_function("q4_qkv_kparallel").unwrap();
        let columns = 96u32;
        let qrows = 5u32;
        let krows = 3u32;
        let make = |rows: u32, seed: u32| {
            let mut bytes = Vec::new();
            for b in 0..rows * columns / 32 {
                bytes.extend([0u16, 1, 0x3800, 0xbc00][b as usize % 4].to_le_bytes());
                bytes.extend((0..16).map(|j| (j * 37 + b * 11 + seed) as u8));
            }
            Q4Matrix::new(columns, rows, bytes).unwrap()
        };
        let q = make(qrows, 3);
        let k = make(krows, 9);
        let v = make(krows, 17);
        let qw = stream.clone_htod(q.bytes()).unwrap();
        let kw = stream.clone_htod(k.bytes()).unwrap();
        let vw = stream.clone_htod(v.bytes()).unwrap();
        for shared in [0u32, 1] {
            let x: Vec<f32> = (0..columns)
                .map(|i| (i % 17) as f32 * 0.125 - 0.5)
                .collect();
            let input = stream.clone_htod(&x).unwrap();
            let mut qo = stream.alloc_zeros::<f32>(qrows as usize).unwrap();
            let mut ko = stream.alloc_zeros::<f32>(krows as usize).unwrap();
            let mut vo = stream.clone_htod(&vec![123.0f32; krows as usize]).unwrap();
            let mut launch = stream.launch_builder(&kernel);
            launch
                .arg(&qw)
                .arg(&kw)
                .arg(&vw)
                .arg(&input)
                .arg(&mut qo)
                .arg(&mut ko)
                .arg(&mut vo)
                .arg(&columns)
                .arg(&qrows)
                .arg(&krows)
                .arg(&shared);
            // SAFETY: one complete four-warp CTA per output row; separate Q/K/V
            // buffers match their validated matrices; shared mode excludes V writes.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (qrows + krows + if shared == 0 { krows } else { 0 }, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
            let qa = stream.clone_dtoh(&qo).unwrap();
            let ka = stream.clone_dtoh(&ko).unwrap();
            let va = stream.clone_dtoh(&vo).unwrap();
            stream.synchronize().unwrap();
            compare(&q.reference_matvec(&x).unwrap(), &qa).unwrap();
            compare(&k.reference_matvec(&x).unwrap(), &ka).unwrap();
            if shared == 0 {
                compare(&v.reference_matvec(&x).unwrap(), &va).unwrap();
            } else {
                assert_eq!(va, vec![123.0; krows as usize]);
            }
        }
    }
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn gate_up_handles_odd_rows_and_short_input() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let ptx = compile_ptx_with_opts(
            include_str!("../../../kernels/ada/q4_matvec.cu"),
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let kernel = module.load_function("q4_gate_up").unwrap();
        for rows in [1u32, 5, 8, 9] {
            for columns in [32u32, 96, 384] {
                let make = |seed: u32| {
                    let mut bytes = Vec::new();
                    for block in 0..rows * columns / 32 {
                        bytes.extend([0x3800u16, 0xbc00, 1, 0][block as usize % 4].to_le_bytes());
                        bytes.extend((0..16).map(|i| (i * 37 + block * 11 + seed) as u8));
                    }
                    Q4Matrix::new(columns, rows, bytes).unwrap()
                };
                let gate = make(3);
                let up = make(19);
                let x: Vec<f32> = (0..columns)
                    .map(|i| (i * 17 % 97) as f32 / 48.0 - 1.0)
                    .collect();
                let wg = stream.clone_htod(gate.bytes()).unwrap();
                let wu = stream.clone_htod(up.bytes()).unwrap();
                let input = stream.clone_htod(&x).unwrap();
                let mut g = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let mut u = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let mut launch = stream.launch_builder(&kernel);
                launch
                    .arg(&wg)
                    .arg(&wu)
                    .arg(&input)
                    .arg(&mut g)
                    .arg(&mut u)
                    .arg(&columns)
                    .arg(&rows);
                // SAFETY: validated matrices, distinct output buffers, four complete warps;
                // rows outside the matrix retire as complete warps.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: ((rows * 2).div_ceil(4), 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();
                let g = stream.clone_dtoh(&g).unwrap();
                let u = stream.clone_dtoh(&u).unwrap();
                stream.synchronize().unwrap();
                compare(&gate.reference_matvec(&x).unwrap(), &g).unwrap();
                compare(&up.reference_matvec(&x).unwrap(), &u).unwrap();
            }
        }
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn synthetic_gpu_handles_partial_warp_group_and_multiple_blocks() {
        let mut bytes = Vec::new();
        for i in 0..5 * 3 {
            let scale = if i % 2 == 0 { 0x3800_u16 } else { 0xbc00_u16 };
            bytes.extend(scale.to_le_bytes());
            bytes.extend((0..16).map(|j| ((j + i) % 16) as u8 | (((15 - j + i) % 16) as u8) << 4));
        }
        let m = Q4Matrix::new(96, 5, bytes).unwrap();
        assert!(check(&m).unwrap().max_tolerance_ratio <= 1.0);
    }
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn kparallel_handles_short_uneven_and_long_input_dimensions() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let ptx = compile_ptx_with_opts(
            include_str!("../../../kernels/ada/q4_matvec.cu"),
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let kernel = module.load_function("q4_matvec_kparallel").unwrap();
        for columns in [32u32, 96, 384, 15360] {
            let rows = 5u32;
            let mut bytes = Vec::new();
            for block in 0..rows * columns / 32 {
                let scale = [0x3800u16, 0xbc00, 1, 0][block as usize % 4];
                bytes.extend(scale.to_le_bytes());
                bytes.extend((0..16).map(|i| (i * 37 + block * 11) as u8));
            }
            let matrix = Q4Matrix::new(columns, rows, bytes).unwrap();
            let weights = stream.clone_htod(matrix.bytes()).unwrap();
            let mut out = stream.alloc_zeros::<f32>(rows as usize).unwrap();
            for amplitude in [0.0f32, 0.125, 1.0] {
                let input: Vec<f32> = (0..columns)
                    .map(|i| ((i * 17 % 97) as f32 / 48.0 - 1.0) * amplitude)
                    .collect();
                let expected = matrix.reference_matvec(&input).unwrap();
                let input = stream.clone_htod(&input).unwrap();
                let mut launch = stream.launch_builder(&kernel);
                launch
                    .arg(&weights)
                    .arg(&input)
                    .arg(&mut out)
                    .arg(&columns)
                    .arg(&rows);
                // SAFETY: one full 128-thread CTA per row, validated matrix and
                // disjoint buffers; all four warps reach the shared reduction.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (rows, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();
                let actual = stream.clone_dtoh(&out).unwrap();
                stream.synchronize().unwrap();
                compare(&expected, &actual).unwrap();
            }
        }
    }
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn thread_major_matches_native_bits_and_times_down_shape() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let ptx = compile_ptx_with_opts(
            format!(
                "{}
{}",
                include_str!("../../../kernels/ada/q4_matvec.cu"),
                include_str!("../../../kernels/ada/q4_thread_major.cu")
            ),
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let shapes = [(32u32, 5u32), (96, 5), (384, 7), (3840, 4096), (3840, 30720), (15360, 3840), (8192, 3840)];
        for (columns, rows) in shapes {
            let mut state = 0x1234_5678u32 ^ columns;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            };
            let mut bytes = Vec::with_capacity((rows * columns / 32 * 18) as usize);
            for _ in 0..rows * columns / 32 {
                let r = next();
                // Finite halves in roughly [2^-12, 2^2], signed.
                let scale = (((r >> 16) & 0x8000) | (0x1000 + (r & 0x3fff) % 0x3000)) as u16;
                bytes.extend(scale.to_le_bytes());
                bytes.extend((0..16).map(|_| next() as u8));
            }
            let matrix = Q4Matrix::new(columns, rows, bytes).unwrap();
            let input: Vec<f32> = (0..columns).map(|_| (next() % 65536) as f32 / 32768.0 - 1.0).collect();
            let input = stream.clone_htod(&input).unwrap();
            let copies = (320usize << 20).div_ceil(matrix.bytes().len()).clamp(2, 64);
            let native: Vec<_> = (0..copies).map(|_| stream.clone_htod(matrix.bytes()).unwrap()).collect();
            for (partitions, native_name, tm_name) in
                [(4usize, "q4_matvec_kparallel", "q4_tm_matvec_kparallel"), (1, "q4_matvec", "q4_tm_matvec"), (0, "q4_matvec_kparallel", "q4_h_matvec_kparallel"), (9, "q4_matvec", "q4_h_matvec")]
            {
                let hoisted = partitions == 0 || partitions == 9;
                let partitions = match partitions { 0 => 4, 9 => 1, p => p };
                let (codes, scales) = thread_major(&matrix, partitions);
                let packed: Vec<_> = (0..copies)
                    .map(|_| (stream.clone_htod(&codes).unwrap(), stream.clone_htod(&scales).unwrap()))
                    .collect();
                let grid = if partitions == 4 { rows } else { rows.div_ceil(4) };
                let config = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                let nk = module.load_function(native_name).unwrap();
                let tk = module.load_function(tm_name).unwrap();
                let mut a = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let mut b = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                // Distinct copies back to back: weights stream from DRAM as in decode.
                let mut time = |tm: bool, out: &mut cudarc::driver::CudaSlice<f32>| {
                    let mut samples = Vec::new();
                    for rep in 0..6 {
                        let e0 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                        for copy in 0..copies {
                            let mut l = stream.launch_builder(if tm { &tk } else { &nk });
                            if tm && hoisted {
                                l.arg(&native[copy]);
                            } else if tm {
                                l.arg(&packed[copy].0).arg(&packed[copy].1);
                            } else {
                                l.arg(&native[copy]);
                            }
                            l.arg(&input).arg(&mut *out).arg(&columns).arg(&rows);
                            // SAFETY: validated matrix, packed buffers sized by thread_major,
                            // full 128-thread CTAs, disjoint output.
                            unsafe { l.launch(config) }.unwrap();
                        }
                        let e1 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                        if rep > 0 {
                            samples.push(e0.elapsed_ms(&e1).unwrap() / copies as f32);
                        }
                    }
                    samples.sort_by(f32::total_cmp);
                    samples[samples.len() / 2]
                };
                let tn = time(false, &mut a);
                let tt = time(true, &mut b);
                let (a, b) = (stream.clone_dtoh(&a).unwrap(), stream.clone_dtoh(&b).unwrap());
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(x.to_bits(), y.to_bits(), "{columns}x{rows} P{partitions} row {i}: {x} vs {y}");
                }
                let gbs = |t: f32| matrix.bytes().len() as f32 / t / 1e6;
                println!("TM {columns}x{rows} P{partitions}{}:", if hoisted {" hoisted"} else {""}); println!("    native {tn:.4} ms ({:.0} GB/s), thread-major {tt:.4} ms ({:.0} GB/s), x{:.2}", gbs(tn), gbs(tt), tn / tt);
            }
        }
    }
}
