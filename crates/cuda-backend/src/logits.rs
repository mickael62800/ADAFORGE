use super::failure;
use adaforge_application::{DiagnosticError, LogitsReport, OutputHead};
use cudarc::{
    driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::sync::Arc;
fn tiled_q6() -> bool {
    std::env::var("ADAFORGE_EXPERIMENTAL_Q6_TILES").as_deref() == Ok("1")
}
fn pack_q6_tiles(bytes: &[u8], columns: u32, rows: u32) -> Vec<u8> {
    let blocks = columns as usize / 256;
    let mut packed = vec![0; (rows as usize).div_ceil(4) * blocks * 840];
    for row in 0..rows as usize {
        for block in 0..blocks {
            let src = &bytes[(row * blocks + block) * 210..][..210];
            let dst = &mut packed[((row / 4) * blocks + block) * 840..][..840];
            for i in 0..208 {
                dst[i * 4 + row % 4] = src[i];
            }
            dst[832 + (row % 4) * 2..834 + (row % 4) * 2].copy_from_slice(&src[208..210]);
        }
    }
    packed
}
fn upload_weights(
    stream: &Arc<CudaStream>,
    head: &OutputHead<'_>,
    tiled: bool,
) -> Result<CudaSlice<u8>, DiagnosticError> {
    if tiled {
        stream.clone_htod(&pack_q6_tiles(
            head.matrix.bytes(),
            head.matrix.columns(),
            head.matrix.rows(),
        ))
    } else {
        stream.clone_htod(head.matrix.bytes())
    }
    .map_err(|e| failure("Poids Q6_K", e))
}
/// Default head layout: lossless repack (see q6_packed.cu). `ADAFORGE_Q6_PACKED=0` keeps GGUF bytes.
fn packed_q6() -> bool {
    std::env::var("ADAFORGE_Q6_PACKED").as_deref() != Ok("0")
}
/// Returns (codes: 208 bytes per block, d halves per block).
pub(crate) fn pack_q6_lanes(bytes: &[u8]) -> (Vec<u8>, Vec<u16>) {
    let blocks = bytes.len() / 210;
    let mut codes = vec![0u8; blocks * 208];
    let mut d = vec![0u16; blocks];
    for block in 0..blocks {
        let src = &bytes[block * 210..][..210];
        let dst = &mut codes[block * 208..][..208];
        for (o, &byte) in src[..192].iter().enumerate() {
            let (k, lane) = (o / 16, o % 16);
            dst[(k / 4) * 64 + lane * 4 + k % 4] = byte;
        }
        dst[192..208].copy_from_slice(&src[192..208]);
        d[block] = u16::from_le_bytes([src[208], src[209]]);
    }
    (codes, d)
}
fn projection_rows() -> u32 {
    match std::env::var("ADAFORGE_EXPERIMENTAL_Q6_ROWS").as_deref() {
        Ok("8") => 8,
        _ => 4,
    }
}
pub(crate) fn check(
    stream: Arc<CudaStream>,
    head: OutputHead<'_>,
    input: &CudaSlice<f32>,
    cpu_input: &[f32],
    position: u32,
) -> Result<LogitsReport, DiagnosticError> {
    if input.len() != head.matrix.columns() as usize
        || cpu_input.len() != input.len()
        || head.norm.weights().len() != input.len()
        || input.context() != stream.context()
        || !head.cap.is_finite()
        || head.cap <= 0.0
    {
        return Err(DiagnosticError("Entrée de tête incompatible".into()));
    }
    let tiled = tiled_q6();
    let width = head.matrix.columns();
    let rows = head.matrix.rows();
    let source = format!(
        "{}\n{}\n{}\n{}",
        include_str!("../../../kernels/ada/embedding_q6.cu"),
        include_str!("../../../kernels/ada/q6_matvec.cu"),
        include_str!("../../../kernels/ada/q6_tiled.cu"),
        include_str!("../../../kernels/ada/rms_norm.cu")
    );
    let ptx = compile_ptx_with_opts(
        source,
        CompileOptions {
            arch: Some("compute_89"),
            ..Default::default()
        },
    )
    .map_err(|e| failure("Compilation logits", e))?;
    let module = stream
        .context()
        .load_module(ptx)
        .map_err(|e| failure("Module logits", e))?;
    let rms = module
        .load_function("rms_norm")
        .map_err(|e| failure("RMS finale", e))?;
    let project = module
        .load_function(if tiled {
            "q6_matvec_tiled"
        } else {
            "q6_matvec"
        })
        .map_err(|e| failure("Projection logits", e))?;
    let scales = stream
        .clone_htod(head.norm.weights())
        .map_err(|e| failure("Poids norme finale", e))?;
    let weights = upload_weights(&stream, &head, tiled)?;
    let mut normalized = stream
        .alloc_zeros::<f32>(width as usize)
        .map_err(|e| failure("Norme finale", e))?;
    let mut raw = stream
        .alloc_zeros::<f32>(rows as usize)
        .map_err(|e| failure("Logits bruts", e))?;
    let mut capped = stream
        .alloc_zeros::<f32>(rows as usize)
        .map_err(|e| failure("Logits finaux", e))?;
    let eps = head.norm.epsilon();
    let cap = head.cap;
    let mut launch = stream.launch_builder(&rms);
    launch
        .arg(input)
        .arg(&scales)
        .arg(&mut normalized)
        .arg(&width)
        .arg(&eps);
    // SAFETY: head geometry validated before CUDA; distinct buffers of width elements.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| failure("RMS finale", e))?;
    let launch_rows = if tiled { 4 } else { projection_rows() };
    let mut launch = stream.launch_builder(&project);
    launch
        .arg(&weights)
        .arg(&normalized)
        .arg(&mut raw)
        .arg(&mut capped)
        .arg(&width)
        .arg(&rows)
        .arg(&cap);
    // SAFETY: Q6_K validated storage, one warp per row, bounded rows, distinct outputs.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows.div_ceil(launch_rows), 1, 1),
            block_dim: (launch_rows * 32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| failure("Logits Q6_K", e))?;
    let actual_norm = stream
        .clone_dtoh(&normalized)
        .map_err(|e| failure("Lecture norme finale", e))?;
    let actual_raw = stream
        .clone_dtoh(&raw)
        .map_err(|e| failure("Lecture logits", e))?;
    let actual_cap = stream
        .clone_dtoh(&capped)
        .map_err(|e| failure("Lecture softcap", e))?;
    stream
        .synchronize()
        .map_err(|e| failure("Synchronisation logits", e))?;
    let expected_norm = head.norm.reference(cpu_input).map_err(DiagnosticError)?;
    let (norm_error, norm_ratio) = super::matvec::compare(&expected_norm, &actual_norm)?;
    let norm32 = expected_norm.iter().map(|&v| v as f32).collect::<Vec<_>>();
    let expected_raw = head.matrix.reference(&norm32).map_err(DiagnosticError)?;
    let (raw_error, raw_ratio) = super::matvec::compare(&expected_raw, &actual_raw)?;
    let expected_cap = expected_raw
        .iter()
        .map(|&v| adaforge_domain::softcap(v as f32, cap).map(f64::from))
        .collect::<Result<Vec<_>, _>>()
        .map_err(DiagnosticError)?;
    let (capped_error, capped_ratio) = super::matvec::compare(&expected_cap, &actual_cap)?;
    Ok(LogitsReport {
        vocabulary: rows,
        position,
        norm_error,
        norm_ratio,
        raw_error,
        raw_ratio,
        capped_error,
        capped_ratio,
    })
}

/// Tête et table liée conservées pendant toute une génération.
pub(crate) struct ResidentHead {
    selection: super::greedy::GpuGreedy,
    batch_project: Vec<cudarc::driver::CudaFunction>,
    /// Rows handled by each warp of the projection kernel (2 for the dual kernel).
    rows_per_warp: u32,
    batch_normalized: Option<CudaSlice<f32>>,
    batch_raw: Option<CudaSlice<f32>>,
    stream: Arc<CudaStream>,
    rms: cudarc::driver::CudaFunction,
    project: cudarc::driver::CudaFunction,
    lookup: cudarc::driver::CudaFunction,
    lookup_dynamic: cudarc::driver::CudaFunction,
    weights: CudaSlice<u8>,
    dscales: Option<CudaSlice<u16>>,
    scales: CudaSlice<f32>,
    normalized: CudaSlice<f32>,
    raw: CudaSlice<f32>,
    capped: CudaSlice<f32>,
    width: u32,
    rows: u32,
    eps: f32,
    cap: f32,
    projection_rows: u32,
}
impl ResidentHead {
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        head: OutputHead<'_>,
    ) -> Result<Self, DiagnosticError> {
        let tiled = tiled_q6();
        let packed = !tiled && packed_q6();
        let width = head.matrix.columns();
        let rows = head.matrix.rows();
        if head.norm.weights().len() != width as usize || !head.cap.is_finite() || head.cap <= 0.0 {
            return Err(DiagnosticError("Tête incompatible".into()));
        }
        let source = format!(
            "{}\n{}\n{}\n{}",
            include_str!("../../../kernels/ada/embedding_q6.cu"),
            include_str!("../../../kernels/ada/q6_matvec.cu"),
            include_str!("../../../kernels/ada/q6_tiled.cu"),
            include_str!("../../../kernels/ada/rms_norm.cu")
        );
        let source = format!("{source}
{}", include_str!("../../../kernels/ada/q6_packed.cu"));
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| failure("Compilation tête résidente", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| failure("Module tête résidente", e))?;
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| failure("Kernel tête résidente", e))
        };
        let rms = load("rms_norm")?;
        // Greedy consumes raw logits. The diagnostic above still checks softcap.
        let raw = std::env::var("ADAFORGE_EXPERIMENTAL_Q6_RAW").as_deref() != Ok("0");
        let project = load(match (tiled, raw) {
            _ if packed && raw => "q6_matvec_raw_packed_dual",
            _ if packed => "q6_matvec_packed",
            (true, true) => "q6_matvec_raw_tiled",
            (true, false) => "q6_matvec_tiled",
            (false, true) => "q6_matvec_raw",
            (false, false) => "q6_matvec",
        })?;
        let lookup = load(if packed {
            "lookup_q6_packed"
        } else if tiled {
            "lookup_q6_tiled"
        } else {
            "lookup_q6"
        })?;
        let lookup_dynamic = load(if packed {
            "lookup_dynamic_packed"
        } else if tiled {
            "lookup_dynamic_tiled"
        } else {
            "lookup_dynamic"
        })?;
        // The lossless repack is computed once for both buffers.
        let (weights, dscales) = if packed {
            let (codes, d) = pack_q6_lanes(head.matrix.bytes());
            (
                stream
                    .clone_htod(&codes)
                    .map_err(|e| failure("Poids Q6_K", e))?,
                Some(
                    stream
                        .clone_htod(&d)
                        .map_err(|e| failure("Échelles Q6_K", e))?,
                ),
            )
        } else {
            (upload_weights(&stream, &head, tiled)?, None)
        };
        Ok(Self {
            selection: super::greedy::GpuGreedy::new(stream.clone(), rows)?,
            // Per batch size, the faster of two bit-identical variants
            // (measured: the two-pass kernel wins at 4 and 8 tokens).
            batch_project: if packed {
                (1..=8)
                    .map(|b| {
                        load(&if b == 4 || b == 8 {
                            format!("q6_matvec_raw_packed2_b{b}")
                        } else {
                            format!("q6_matvec_raw_packed_b{b}")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            },
            batch_normalized: None,
            batch_raw: None,
            rows_per_warp: if packed && raw { 2 } else { 1 },
            rms,
            project,
            lookup,
            lookup_dynamic,
            width,
            rows,
            eps: head.norm.epsilon(),
            cap: head.cap,
            projection_rows: if tiled { 4 } else { projection_rows() },
            weights,
            dscales,
            scales: stream
                .clone_htod(head.norm.weights())
                .map_err(|e| failure("Norme résidente", e))?,
            normalized: stream
                .alloc_zeros(width as usize)
                .map_err(|e| failure("Buffer norme", e))?,
            raw: stream
                .alloc_zeros(rows as usize)
                .map_err(|e| failure("Buffer logits", e))?,
            capped: stream
                .alloc_zeros(rows as usize)
                .map_err(|e| failure("Buffer softcap", e))?,
            stream,
        })
    }
    pub(crate) fn embed(
        &self,
        token: u32,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), DiagnosticError> {
        if token >= self.rows
            || output.len() != self.width as usize
            || output.context() != self.stream.context()
        {
            return Err(DiagnosticError("Lookup embeddings invalide".into()));
        }
        let mut launch = self.stream.launch_builder(&self.lookup);
        launch
            .arg(&self.weights);
        if let Some(d) = &self.dscales {
            launch.arg(d);
        }
        launch
            .arg(output)
            .arg(&self.width)
            .arg(&token);
        // SAFETY: token bounds and output length checked; validated Q6_K table, bounded channel writes.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.width.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Lookup Q6_K", e))?;
        Ok(())
    }
    pub(crate) fn embed_graph(
        &self,
        state: &CudaSlice<u32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), DiagnosticError> {
        let mut launch = self.stream.launch_builder(&self.lookup_dynamic);
        launch
            .arg(&self.weights);
        if let Some(d) = &self.dscales {
            launch.arg(d);
        }
        launch
            .arg(output)
            .arg(&self.width)
            .arg(state);
        // SAFETY: private session graph; token and output bounds checked before every replay.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.width.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Embedding graphe", e))?;
        Ok(())
    }
    fn project_gpu(&mut self, input: &CudaSlice<f32>) -> Result<(), DiagnosticError> {
        if input.len() != self.width as usize || input.context() != self.stream.context() {
            return Err(DiagnosticError("Entrée tête invalide".into()));
        }
        let mut launch = self.stream.launch_builder(&self.rms);
        launch
            .arg(input)
            .arg(&self.scales)
            .arg(&mut self.normalized)
            .arg(&self.width)
            .arg(&self.eps);
        // SAFETY: validated dimensions, one 256-thread block, separate buffers.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Norme finale résidente", e))?;
        let mut launch = self.stream.launch_builder(&self.project);
        launch
            .arg(&self.weights);
        if let Some(d) = &self.dscales {
            launch.arg(d);
        }
        launch
            .arg(&self.normalized)
            .arg(&mut self.raw)
            .arg(&mut self.capped)
            .arg(&self.width)
            .arg(&self.rows)
            .arg(&self.cap);
        // SAFETY: validated table dimensions; one warp per bounded row, disjoint output buffers.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.rows.div_ceil(self.projection_rows * self.rows_per_warp), 1, 1),
                block_dim: (self.projection_rows * 32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Projection résidente", e))?;
        Ok(())
    }
    pub(crate) fn supports_batch(&self) -> bool {
        !self.batch_project.is_empty()
    }
    /// Greedy ids for `n` hidden states [n][width], reading the table once.
    /// Each id equals `select` on the corresponding row (bit-identical logits).
    pub(crate) fn select_batch(
        &mut self,
        input: &CudaSlice<f32>,
        n: usize,
    ) -> Result<Vec<u32>, DiagnosticError> {
        let width = self.width as usize;
        let per_launch = self.batch_project.len();
        if n == 0 || n > 2 * per_launch || input.len() < n * width {
            return Err(DiagnosticError("Lot tête invalide".into()));
        }
        if self.batch_normalized.is_none() {
            self.batch_normalized = Some(
                self.stream
                    .alloc_zeros(16 * width)
                    .map_err(|e| failure("Buffer norme lot", e))?,
            );
            self.batch_raw = Some(
                self.stream
                    .alloc_zeros(16 * self.rows as usize)
                    .map_err(|e| failure("Buffer logits lot", e))?,
            );
        }
        let normalized = self.batch_normalized.as_mut().expect("allocated");
        let raw = self.batch_raw.as_mut().expect("allocated");
        let mut launch = self.stream.launch_builder(&self.rms);
        launch
            .arg(input)
            .arg(&self.scales)
            .arg(&mut *normalized)
            .arg(&self.width)
            .arg(&self.eps);
        // SAFETY: n <= 16 vectors of width in input and normalized, one block each.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failure("Norme finale lot", e))?;
        // Balanced launches of at most 8 tokens (the table is read once each).
        let rows = self.rows as usize;
        let launches = n.div_ceil(per_launch);
        let chunk = n.div_ceil(launches);
        let mut first = 0;
        while first < n {
            let m = chunk.min(n - first);
            let x = normalized.slice(first * width..);
            let mut y = raw.slice_mut(first * rows..);
            let mut launch = self.stream.launch_builder(&self.batch_project[m - 1]);
            launch
                .arg(&self.weights)
                .arg(self.dscales.as_ref().expect("packed head"))
                .arg(&x)
                .arg(&mut y)
                .arg(&self.width)
                .arg(&self.rows);
            // SAFETY: packed table; the views hold m*width inputs and m*rows logits.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (self.rows.div_ceil(4), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| failure("Projection lot", e))?;
            first += m;
        }
        self.selection.select_many(raw, n)
    }
    pub(crate) fn select(&mut self, input: &CudaSlice<f32>) -> Result<u32, DiagnosticError> {
        self.project_gpu(input)?;
        self.selection.select(&self.raw)
    }
    #[cfg(test)]
    pub(crate) fn run(&mut self, input: &CudaSlice<f32>) -> Result<Vec<f32>, DiagnosticError> {
        self.project_gpu(input)?;
        let result = self
            .stream
            .clone_dtoh(&self.raw)
            .map_err(|e| failure("Lecture référence greedy", e))?;
        self.stream
            .synchronize()
            .map_err(|e| failure("Synchronisation référence greedy", e))?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn raw_projection_matches_full_bits_and_skips_capped_writes() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let source = format!(
            "{}\n{}\n{}",
            include_str!("../../../kernels/ada/embedding_q6.cu"),
            include_str!("../../../kernels/ada/q6_matvec.cu"),
            include_str!("../../../kernels/ada/q6_tiled.cu")
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
        for columns in [256u32, 768, 3840] {
            let rows = 9u32;
            let mut bytes = vec![0u8; rows as usize * (columns as usize / 256) * 210];
            for (n, block) in bytes.chunks_mut(210).enumerate() {
                for (i, v) in block[..208].iter_mut().enumerate() {
                    *v = (i * 37 + n * 13) as u8;
                }
                block[208..]
                    .copy_from_slice(&([0u16, 0x8000, 1, 0x2800, 0xa800][n % 5]).to_le_bytes());
            }
            let weights = stream.clone_htod(&bytes).unwrap();
            let packed = pack_q6_tiles(&bytes, columns, rows);
            let tiled_weights = stream.clone_htod(&packed).unwrap();
            let input = stream
                .clone_htod(
                    &(0..columns)
                        .map(|i| (i % 19) as f32 / 9.0 - 1.0)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut outputs = Vec::new();
            for name in [
                "q6_matvec",
                "q6_matvec_raw",
                "q6_matvec_tiled",
                "q6_matvec_raw_tiled",
            ] {
                let kernel = module.load_function(name).unwrap();
                let mut raw = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let mut capped = stream.clone_htod(&vec![12345.0f32; rows as usize]).unwrap();
                let cap = 30.0f32;
                let mut launch = stream.launch_builder(&kernel);
                launch
                    .arg(if name.ends_with("tiled") {
                        &tiled_weights
                    } else {
                        &weights
                    })
                    .arg(&input)
                    .arg(&mut raw)
                    .arg(&mut capped)
                    .arg(&columns)
                    .arg(&rows)
                    .arg(&cap);
                // SAFETY: validated synthetic Q6 storage, 4 warps per block and bounded partial group.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (rows.div_ceil(4), 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();
                let values = stream.clone_dtoh(&raw).unwrap();
                let caps = stream.clone_dtoh(&capped).unwrap();
                stream.synchronize().unwrap();
                outputs.push(values.into_iter().map(f32::to_bits).collect::<Vec<_>>());
                if name.contains("raw") {
                    assert_eq!(caps, vec![12345.0; rows as usize]);
                }
            }
            for result in &outputs[1..] {
                assert_eq!(&outputs[0], result, "columns={columns}");
            }
        }
    }
    #[test]
    #[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
    fn q6_head_signed_scales_multiple_blocks_and_partial_row_group() {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let mut bytes = Vec::new();
        for block in 0..10 {
            let mut b = vec![0; 210];
            for (i, v) in b[..192].iter_mut().enumerate() {
                *v = (i * 37 + block * 11) as u8;
            }
            for i in 0..16 {
                b[192 + i] = (i as i8 - 8) as u8;
            }
            let scale = if block == 0 {
                1u16
            } else if block % 2 == 0 {
                0xa800
            } else {
                0x2800
            };
            b[208..].copy_from_slice(&scale.to_le_bytes());
            bytes.extend(b);
        }
        let matrix = adaforge_domain::Q6Matrix::new(512, 5, bytes).unwrap();
        let norm = adaforge_domain::RmsNorm::new(
            (0..512)
                .map(|i| if i % 2 == 0 { 0.75 } else { -1.25 })
                .collect(),
            1e-6,
        )
        .unwrap();
        let x = (0..512)
            .map(|i| (i % 17) as f32 / 8.0 - 1.0)
            .collect::<Vec<_>>();
        let input = stream.clone_htod(&x).unwrap();
        let report = check(
            stream.clone(),
            OutputHead {
                matrix: &matrix,
                norm: &norm,
                cap: 1.5,
            },
            &input,
            &x,
            0,
        )
        .unwrap();
        let mut resident = ResidentHead::new(
            stream.clone(),
            OutputHead {
                matrix: &matrix,
                norm: &norm,
                cap: 1.5,
            },
        )
        .unwrap();
        let actual = resident.run(&input).unwrap();
        assert_eq!(
            resident.select(&input).unwrap(),
            adaforge_domain::greedy(&actual).unwrap()
        );
        let n = norm
            .reference(&x)
            .unwrap()
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        let expected = matrix.reference(&n).unwrap();
        super::super::matvec::compare(&expected, &actual).unwrap();
        assert_eq!(
            adaforge_domain::greedy(&actual).unwrap(),
            adaforge_domain::greedy(&expected.iter().map(|&v| v as f32).collect::<Vec<_>>())
                .unwrap()
        );
        let mut embedding = stream.alloc_zeros::<f32>(512).unwrap();
        for id in [4u32, 0, 2, 4] {
            resident.embed(id, &mut embedding).unwrap();
            let actual = stream.clone_dtoh(&embedding).unwrap();
            stream.synchronize().unwrap();
            let start = id as usize * 420;
            let batch = adaforge_domain::EmbeddingBatch::new(
                512,
                5,
                vec![id],
                matrix.bytes()[start..start + 420].to_vec(),
            )
            .unwrap();
            assert_eq!(actual, batch.reference(true));
        }
        assert!(resident.embed(5, &mut embedding).is_err());
        assert_eq!(report.vocabulary, 5);
        assert!(report.raw_ratio <= 1.0);
        assert!(report.capped_ratio <= 1.0);
        assert!(
            check(
                stream,
                OutputHead {
                    matrix: &matrix,
                    norm: &norm,
                    cap: 0.0
                },
                &input,
                &x,
                0
            )
            .is_err()
        );
    }
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn packed_q6_projection_and_lookup_match_native_bits() {
        use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let source = format!(
            "{}
{}
{}",
            include_str!("../../../kernels/ada/embedding_q6.cu"),
            include_str!("../../../kernels/ada/q6_matvec.cu"),
            include_str!("../../../kernels/ada/q6_packed.cu")
        );
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions { arch: Some("compute_89"), ..Default::default() },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let mut state = 0x9e37_79b9u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for (columns, rows) in [(256u32, 3u32), (768, 9), (3840, 517)] {
            let blocks = (columns / 256 * rows) as usize;
            let mut bytes = vec![0u8; blocks * 210];
            for block in 0..blocks {
                let b = &mut bytes[block * 210..][..210];
                b[..208].iter_mut().for_each(|v| *v = next() as u8);
                let r = next();
                let d = (((r >> 16) & 0x8000) | (0x0400 + (r & 0x3fff) % 0x3800)) as u16;
                b[208..].copy_from_slice(&d.to_le_bytes());
            }
            let (codes, d) = pack_q6_lanes(&bytes);
            let native = stream.clone_htod(&bytes).unwrap();
            let codes = stream.clone_htod(&codes).unwrap();
            let d = stream.clone_htod(&d).unwrap();
            let x: Vec<f32> = (0..columns).map(|_| (next() % 65536) as f32 / 32768.0 - 1.0).collect();
            let x = stream.clone_htod(&x).unwrap();
            let cap = 30.0f32;
            let run = |name: &str, packed: bool| {
                let f = module.load_function(name).unwrap();
                let mut raw = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let mut capped = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let mut l = stream.launch_builder(&f);
                if packed {
                    l.arg(&codes).arg(&d);
                } else {
                    l.arg(&native);
                }
                l.arg(&x).arg(&mut raw).arg(&mut capped).arg(&columns).arg(&rows).arg(&cap);
                // SAFETY: buffers sized from rows/columns; one warp per bounded row.
                unsafe {
                    l.launch(LaunchConfig { grid_dim: (rows.div_ceil(4), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
                }
                .unwrap();
                (stream.clone_dtoh(&raw).unwrap(), stream.clone_dtoh(&capped).unwrap())
            };
            let a = run("q6_matvec", false);
            let b = run("q6_matvec_packed", true);
            let c = run("q6_matvec_raw_packed", true);
            for i in 0..rows as usize {
                assert_eq!(a.0[i].to_bits(), b.0[i].to_bits(), "raw row {i}");
                assert_eq!(a.1[i].to_bits(), b.1[i].to_bits(), "capped row {i}");
                assert_eq!(a.0[i].to_bits(), c.0[i].to_bits(), "raw-only row {i}");
            }
            for token in [0, rows / 2, rows - 1] {
                let lookup = |name: &str, packed: bool| {
                    let f = module.load_function(name).unwrap();
                    let mut out = stream.alloc_zeros::<f32>(columns as usize).unwrap();
                    let mut l = stream.launch_builder(&f);
                    if packed {
                        l.arg(&codes).arg(&d);
                    } else {
                        l.arg(&native);
                    }
                    l.arg(&mut out).arg(&columns).arg(&token);
                    // SAFETY: token < rows, bounded channel writes.
                    unsafe {
                        l.launch(LaunchConfig { grid_dim: (columns.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                    }
                    .unwrap();
                    stream.clone_dtoh(&out).unwrap()
                };
                let (n, p) = (lookup("lookup_q6", false), lookup("lookup_q6_packed", true));
                assert!(n.iter().zip(&p).all(|(a, b)| a.to_bits() == b.to_bits()), "lookup token {token}");
            }
        }
    }
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn batched_q6_head_matches_single_bits_and_amortizes() {
        use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg, sys};
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let source = format!(
            "{}
{}
{}",
            include_str!("../../../kernels/ada/embedding_q6.cu"),
            include_str!("../../../kernels/ada/q6_matvec.cu"),
            include_str!("../../../kernels/ada/q6_packed.cu")
        );
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions { arch: Some("compute_89"), ..Default::default() },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let mut state = 0x3c6e_f372u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for (columns, rows) in [(768u32, 37u32), (3840, 262144)] {
            let blocks = (columns / 256 * rows) as usize;
            let mut bytes = vec![0u8; blocks * 210];
            for block in 0..blocks {
                let b = &mut bytes[block * 210..][..210];
                b[..208].iter_mut().for_each(|v| *v = next() as u8);
                let r = next();
                let d = (((r >> 16) & 0x8000) | (0x0400 + (r & 0x3fff) % 0x3800)) as u16;
                b[208..].copy_from_slice(&d.to_le_bytes());
            }
            let (codes, d) = pack_q6_lanes(&bytes);
            drop(bytes);
            let (codes, d) = (stream.clone_htod(&codes).unwrap(), stream.clone_htod(&d).unwrap());
            let x: Vec<f32> = (0..columns * 8).map(|_| (next() % 65536) as f32 / 32768.0 - 1.0).collect();
            let xs = stream.clone_htod(&x).unwrap();
            let cfg = LaunchConfig { grid_dim: (rows.div_ceil(4), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            let single = module.load_function("q6_matvec_raw_packed").unwrap();
            let mut expected = Vec::new();
            for t in 0..8usize {
                let xt = stream.clone_htod(&x[t * columns as usize..(t + 1) * columns as usize]).unwrap();
                let mut raw = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let mut capped = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let cap = 30.0f32;
                let mut l = stream.launch_builder(&single);
                l.arg(&codes).arg(&d).arg(&xt).arg(&mut raw).arg(&mut capped).arg(&columns).arg(&rows).arg(&cap);
                // SAFETY: buffers sized from rows/columns; one warp per bounded row.
                unsafe { l.launch(cfg) }.unwrap();
                expected.extend(stream.clone_dtoh(&raw).unwrap());
            }
            let mut out = stream.alloc_zeros::<f32>(rows as usize * 8).unwrap();
            let mut report = String::new();
            for n in 1u32..=8 {
                let f = module.load_function(&format!("q6_matvec_raw_packed_b{n}")).unwrap();
                let mut best = f32::MAX;
                for _ in 0..3 {
                    let e0 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    let mut l = stream.launch_builder(&f);
                    l.arg(&codes).arg(&d).arg(&xs).arg(&mut out).arg(&columns).arg(&rows);
                    // SAFETY: x holds 8 tokens, out 8*rows floats.
                    unsafe { l.launch(cfg) }.unwrap();
                    let e1 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    best = best.min(e0.elapsed_ms(&e1).unwrap());
                }
                let got = stream.clone_dtoh(&out).unwrap();
                for i in 0..(n * rows) as usize {
                    assert_eq!(got[i].to_bits(), expected[i].to_bits(), "{columns}x{rows} n={n} i={i}");
                }
                report += &format!(" n{n}={best:.3}ms");
                let f2 = module.load_function(&format!("q6_matvec_raw_packed2_b{n}")).unwrap();
                let mut best2 = f32::MAX;
                for _ in 0..3 {
                    let e0 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    let mut l = stream.launch_builder(&f2);
                    l.arg(&codes).arg(&d).arg(&xs).arg(&mut out).arg(&columns).arg(&rows);
                    // SAFETY: x holds 8 tokens, out 8*rows floats.
                    unsafe { l.launch(cfg) }.unwrap();
                    let e1 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    best2 = best2.min(e0.elapsed_ms(&e1).unwrap());
                }
                let got = stream.clone_dtoh(&out).unwrap();
                for i in 0..(n * rows) as usize {
                    assert_eq!(got[i].to_bits(), expected[i].to_bits(), "two-pass {columns}x{rows} n={n} i={i}");
                }
                report += &format!("/{best2:.3}");
            }
            println!("Q6BATCH {columns}x{rows}:{report}");
        }
    }
    #[test]
    #[ignore = "Requires RTX 4090 and CUDA"]
    fn dual_row_q6_head_matches_bits_and_times() {
        use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg, sys};
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let source = format!(
            "{}
{}
{}",
            include_str!("../../../kernels/ada/embedding_q6.cu"),
            include_str!("../../../kernels/ada/q6_matvec.cu"),
            include_str!("../../../kernels/ada/q6_packed.cu")
        );
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            source,
            cudarc::nvrtc::CompileOptions { arch: Some("compute_89"), ..Default::default() },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let mut state = 0x1f83_d9abu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for (columns, rows) in [(768u32, 37u32), (3840, 262144)] {
            let blocks = (columns / 256 * rows) as usize;
            let mut bytes = vec![0u8; blocks * 210];
            for block in 0..blocks {
                let b = &mut bytes[block * 210..][..210];
                b[..208].iter_mut().for_each(|v| *v = next() as u8);
                let r = next();
                let d = (((r >> 16) & 0x8000) | (0x0400 + (r & 0x3fff) % 0x3800)) as u16;
                b[208..].copy_from_slice(&d.to_le_bytes());
            }
            let (codes, d) = pack_q6_lanes(&bytes);
            drop(bytes);
            let (codes, d) = (stream.clone_htod(&codes).unwrap(), stream.clone_htod(&d).unwrap());
            let x: Vec<f32> = (0..columns).map(|_| (next() % 65536) as f32 / 32768.0 - 1.0).collect();
            let x = stream.clone_htod(&x).unwrap();
            let cap = 30.0f32;
            let run = |name: &str, grid: u32| {
                let f = module.load_function(name).unwrap();
                let mut raw = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let mut capped = stream.alloc_zeros::<f32>(rows as usize).unwrap();
                let mut best = f32::MAX;
                for _ in 0..5 {
                    let e0 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    let mut l = stream.launch_builder(&f);
                    l.arg(&codes).arg(&d).arg(&x).arg(&mut raw).arg(&mut capped).arg(&columns).arg(&rows).arg(&cap);
                    // SAFETY: buffers sized from rows/columns; bounded rows.
                    unsafe { l.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
                    let e1 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    best = best.min(e0.elapsed_ms(&e1).unwrap());
                }
                (stream.clone_dtoh(&raw).unwrap(), best)
            };
            let a = run("q6_matvec_raw_packed", rows.div_ceil(4));
            let b = run("q6_matvec_raw_packed_dual", rows.div_ceil(8));
            assert!(a.0.iter().zip(&b.0).all(|(x, y)| x.to_bits() == y.to_bits()), "dual {columns}x{rows}");
            println!("DUAL {columns}x{rows}: single {:.3}ms dual {:.3}ms", a.1, b.1);
        }
    }
}
