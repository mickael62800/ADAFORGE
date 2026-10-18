use super::failure;
use adaforge_application::{DiagnosticError, RmsNormReport};
use adaforge_domain::RmsNorm;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

pub(crate) fn check(norm: &RmsNorm) -> Result<RmsNormReport, DiagnosticError> {
    let ctx = CudaContext::new(0).map_err(|e| failure("CUDA RMSNorm", e))?;
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
    let width = norm.weights().len() as u32;
    let vectors = 8_u32;
    let mut input = Vec::with_capacity(width as usize * vectors as usize);
    let mut state = 0x4090_u32;
    for case in 0..vectors {
        input.extend((0..width).map(|i| match case {
            0 => 0.0,
            1 => 1.0,
            2 => 1e-8,
            3 => {
                if i % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
            4 => (i % 97) as f32 / 48.0 - 1.0,
            5 => {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & 65535) as f32 / 32768.0 - 1.0
            }
            6 => {
                if i == 0 {
                    10.0
                } else {
                    0.0
                }
            }
            _ => ((i % 17) as f32 - 8.0) * 1e9,
        }));
    }
    let reference = norm.reference(&input).map_err(DiagnosticError)?;
    let ptx = compile_ptx_with_opts(
        include_str!("../../../kernels/ada/rms_norm.cu"),
        CompileOptions {
            arch: Some("compute_89"),
            ..Default::default()
        },
    )
    .map_err(|e| failure("Compilation RMSNorm", e))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| failure("Module RMSNorm", e))?;
    let function = module
        .load_function("rms_norm")
        .map_err(|e| failure("Kernel RMSNorm", e))?;
    let stream = ctx.default_stream();
    let x = stream
        .clone_htod(&input)
        .map_err(|e| failure("Transfert RMSNorm", e))?;
    let weights = stream
        .clone_htod(norm.weights())
        .map_err(|e| failure("Transfert échelles", e))?;
    let mut output = stream
        .alloc_zeros::<f32>(input.len())
        .map_err(|e| failure("Sortie RMSNorm", e))?;
    let config = LaunchConfig {
        grid_dim: (vectors, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let epsilon = norm.epsilon();
    let mut times = Vec::new();
    for repetition in 0..11 {
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| failure("Événement début", e))?;
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&x)
            .arg(&weights)
            .arg(&mut output)
            .arg(&width)
            .arg(&epsilon);
        // SAFETY: bloc de 256 threads conforme au tableau partagé et à la réduction.
        // x et y contiennent vectors*width éléments, weights width, tous vivants
        // jusqu’à synchronisation. Sortie exclusive ; boucles bornées par width.
        unsafe { launch.launch(config) }.map_err(|e| failure("Lancement RMSNorm", e))?;
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| failure("Événement fin", e))?;
        end.synchronize()
            .map_err(|e| failure("Synchronisation RMSNorm", e))?;
        if repetition > 0 {
            times.push(
                start
                    .elapsed_ms(&end)
                    .map_err(|e| failure("Temps RMSNorm", e))?,
            );
        }
    }
    let actual = stream
        .clone_dtoh(&output)
        .map_err(|e| failure("Retour RMSNorm", e))?;
    stream
        .synchronize()
        .map_err(|e| failure("Synchronisation retour", e))?;
    let (max_absolute_error, max_tolerance_ratio) = compare(&reference, &actual)?;
    times.sort_by(f32::total_cmp);
    Ok(RmsNormReport {
        width: width as usize,
        vectors: vectors as usize,
        epsilon,
        max_absolute_error,
        max_tolerance_ratio,
        median_kernel_ms: (times[4] + times[5]) * 0.5,
    })
}
fn compare(reference: &[f64], actual: &[f32]) -> Result<(f64, f64), DiagnosticError> {
    if reference.len() != actual.len() {
        return Err(DiagnosticError("Tailles RMSNorm incompatibles".into()));
    }
    let mut error_max = 0.0_f64;
    let mut ratio_max = 0.0_f64;
    for (i, (&expected, &value)) in reference.iter().zip(actual).enumerate() {
        let error = (f64::from(value) - expected).abs();
        let ratio = error / (2e-5 + 2e-5 * expected.abs());
        if !expected.is_finite() || !value.is_finite() || ratio > 1.0 {
            return Err(DiagnosticError(format!(
                "RMSNorm incorrect indice {i}: CPU {expected}, GPU {value}"
            )));
        }
        error_max = error_max.max(error);
        ratio_max = ratio_max.max(ratio);
    }
    Ok((error_max, ratio_max))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_detects_bad_values() {
        for x in [vec![], vec![f32::NAN], vec![f32::INFINITY], vec![2.0]] {
            assert!(compare(&[1.0], &x).is_err());
        }
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_reduction_handles_small_and_partial_dimensions() {
        for width in [1, 31, 256, 257, 3840] {
            let weights = (0..width).map(|i| (i % 7) as f32 - 3.0).collect();
            let norm = RmsNorm::new(weights, 1e-6).unwrap();
            assert!(check(&norm).unwrap().max_tolerance_ratio <= 1.0);
        }
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn warp_reduction_is_bitwise_equal_to_original_tree() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let old =
            include_str!("../tests/fixtures/rms_norm_before.cu").replace("rms_norm", "legacy_norm");
        let source = format!(
            "{}\n{}",
            include_str!("../../../kernels/ada/rms_norm.cu"),
            old
        );
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let new = module.load_function("rms_norm").unwrap();
        let old = module.load_function("legacy_norm").unwrap();
        for width in [1u32, 7, 32, 64, 256, 512, 3840, 15360] {
            let input = (0..width * 3)
                .map(|i| {
                    if i < width {
                        0.0
                    } else {
                        ((i * 37) % 1023) as f32 / 128.0 - 4.0
                    }
                })
                .collect::<Vec<_>>();
            let weights = (0..width)
                .map(|i| (i % 19) as f32 / 8.0 - 1.0)
                .collect::<Vec<_>>();
            let x = stream.clone_htod(&input).unwrap();
            let w = stream.clone_htod(&weights).unwrap();
            let eps = 1e-6f32;
            let mut outputs = Vec::new();
            for function in [&new, &old] {
                let mut y = stream.alloc_zeros::<f32>(input.len()).unwrap();
                let mut launch = stream.launch_builder(function);
                launch.arg(&x).arg(&w).arg(&mut y).arg(&width).arg(&eps);
                // SAFETY: three contiguous vectors, exclusive output and one 256-thread block per vector.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (3, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();
                let result = stream.clone_dtoh(&y).unwrap();
                stream.synchronize().unwrap();
                outputs.push(result);
            }
            assert!(
                outputs[0]
                    .iter()
                    .zip(&outputs[1])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "width {width}"
            );
        }
    }
}
