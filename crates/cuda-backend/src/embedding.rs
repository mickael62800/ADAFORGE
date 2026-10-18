use super::failure;
use adaforge_application::{DiagnosticError, EmbeddingReport};
use adaforge_domain::EmbeddingBatch;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, sys},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
pub(crate) fn check(batch: &EmbeddingBatch) -> Result<EmbeddingReport, DiagnosticError> {
    let ctx = CudaContext::new(0).map_err(|e| failure("CUDA embeddings", e))?;
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
    let ptx = compile_ptx_with_opts(
        include_str!("../../../kernels/ada/embedding_q6.cu"),
        CompileOptions {
            arch: Some("compute_89"),
            ..Default::default()
        },
    )
    .map_err(|e| failure("Compilation embeddings", e))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| failure("Module embeddings", e))?;
    let function = module
        .load_function("embedding_q6")
        .map_err(|e| failure("Kernel embeddings", e))?;
    let stream = ctx.default_stream();
    let packed = stream
        .clone_htod(batch.bytes())
        .map_err(|e| failure("Transfert embeddings", e))?;
    let elements = batch.token_ids().len() as u32 * batch.width();
    let mut output = stream
        .alloc_zeros::<f32>(elements as usize)
        .map_err(|e| failure("Sortie embeddings", e))?;
    let config = LaunchConfig {
        grid_dim: (elements.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut max_error = 0.0_f64;
    let mut max_ratio = 0.0_f64;
    for scaled in [false, true] {
        let multiplier = if scaled { batch.scale() } else { 1.0 };
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&packed)
            .arg(&mut output)
            .arg(&elements)
            .arg(&multiplier);
        // SAFETY: batch validé, 210 octets pour 256 valeurs, échelles finies,
        // sortie elements f32. Chaque thread écrit un canal borné, aucun alias.
        unsafe { launch.launch(config) }.map_err(|e| failure("Déquantification embeddings", e))?;
        let actual = stream
            .clone_dtoh(&output)
            .map_err(|e| failure("Retour embeddings", e))?;
        stream
            .synchronize()
            .map_err(|e| failure("Synchronisation embeddings", e))?;
        let reference = batch.reference(scaled);
        for (i, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
            let error = (f64::from(a) - f64::from(b)).abs();
            let ratio = error / (1e-6 + 1e-6 * f64::from(b).abs());
            if !a.is_finite() || !b.is_finite() || ratio > 1.0 {
                return Err(DiagnosticError(format!(
                    "Embedding incorrect canal {i}: GPU {a}, CPU {b}"
                )));
            }
            max_error = max_error.max(error);
            max_ratio = max_ratio.max(ratio);
        }
    }
    Ok(EmbeddingReport {
        width: batch.width(),
        vocabulary: batch.vocabulary(),
        token_ids: batch.token_ids().to_vec(),
        scale: batch.scale(),
        max_absolute_error: max_error,
        max_tolerance_ratio: max_ratio,
        verified_values: elements as usize * 2,
    })
}
/// Déquantification GPU conservée vivante pendant la chaîne de calcul.
pub(crate) struct GpuEmbeddings {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    _function: cudarc::driver::CudaFunction,
    _packed: cudarc::driver::CudaSlice<u8>,
    output: cudarc::driver::CudaSlice<f32>,
    width: usize,
    rows: usize,
}
impl GpuEmbeddings {
    pub(crate) fn new(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        batch: &EmbeddingBatch,
    ) -> Result<Self, DiagnosticError> {
        let ptx = compile_ptx_with_opts(
            include_str!("../../../kernels/ada/embedding_q6.cu"),
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| failure("Compilation embeddings chaîne", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| failure("Module embeddings chaîne", e))?;
        let function = module
            .load_function("embedding_q6")
            .map_err(|e| failure("Kernel embeddings chaîne", e))?;
        let packed = stream
            .clone_htod(batch.bytes())
            .map_err(|e| failure("Poids embeddings chaîne", e))?;
        let elements = batch.width() * batch.token_ids().len() as u32;
        let mut output = stream
            .alloc_zeros::<f32>(elements as usize)
            .map_err(|e| failure("Sortie embeddings chaîne", e))?;
        let config = LaunchConfig {
            grid_dim: (elements.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let scale = batch.scale();
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&packed)
            .arg(&mut output)
            .arg(&elements)
            .arg(&scale);
        // SAFETY: batch validé, stockage Q6_K borné, sortie elements f32 ;
        // buffers et fonction gardés vivants dans l’objet, échelles finies.
        unsafe { launch.launch(config) }.map_err(|e| failure("Embedding chaîne", e))?;
        Ok(Self {
            stream,
            _function: function,
            _packed: packed,
            output,
            width: batch.width() as usize,
            rows: batch.token_ids().len(),
        })
    }
    pub(crate) fn copy_row_to(
        &self,
        index: usize,
        destination: &mut cudarc::driver::CudaSlice<f32>,
    ) -> Result<(), DiagnosticError> {
        if index >= self.rows
            || destination.len() != self.width
            || destination.context() != self.stream.context()
        {
            return Err(DiagnosticError(
                "Ligne embeddings ou destination invalide".into(),
            ));
        }
        self.stream
            .memcpy_dtod(
                &self
                    .output
                    .slice(index * self.width..(index + 1) * self.width),
                destination,
            )
            .map_err(|e| failure("Entrée couche depuis embedding GPU", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn gpu_decodes_signed_scales_multiple_rows_and_blocks() {
        let mut bytes = Vec::new();
        for block in 0..6 {
            let mut b = vec![0; 210];
            for (i, v) in b[..192].iter_mut().enumerate() {
                *v = (i * 37 + block * 11) as u8;
            }
            for i in 0..16 {
                b[192 + i] = (i as i8 - 8) as u8;
            }
            let scale = if block == 0 {
                1_u16
            } else if block % 2 == 0 {
                0xb800
            } else {
                0x3800
            };
            b[208..].copy_from_slice(&scale.to_le_bytes());
            bytes.extend(b);
        }
        let batch = EmbeddingBatch::new(512, 10, vec![0, 9, 0], bytes).unwrap();
        let report = check(&batch).unwrap();
        assert_eq!(report.verified_values, 3072);
        assert_eq!(report.max_absolute_error, 0.0);
    }

    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn resident_rows_are_copied_in_order_and_bounds_are_checked() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut bytes = Vec::new();
        for token in [0, 1, 0, 2, 3, 2, 1, 0] {
            let mut block = vec![0; 210];
            for (i, b) in block[..192].iter_mut().enumerate() {
                *b = (i * 17 + token * 29) as u8;
            }
            block[192..208].fill(3);
            block[208..].copy_from_slice(&0x2800_u16.to_le_bytes());
            bytes.extend(block);
        }
        let batch = EmbeddingBatch::new(256, 4, vec![0, 1, 0, 2, 3, 2, 1, 0], bytes).unwrap();
        let reference = batch.reference(true);
        let gpu = GpuEmbeddings::new(stream.clone(), &batch).unwrap();
        let mut destination = stream.alloc_zeros::<f32>(256).unwrap();
        for row in [7, 0, 3, 1, 5, 2] {
            gpu.copy_row_to(row, &mut destination).unwrap();
            let actual = stream.clone_dtoh(&destination).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(actual, &reference[row * 256..(row + 1) * 256]);
        }
        assert!(gpu.copy_row_to(8, &mut destination).is_err());
        let mut wrong = stream.alloc_zeros::<f32>(255).unwrap();
        assert!(gpu.copy_row_to(0, &mut wrong).is_err());
    }
}
