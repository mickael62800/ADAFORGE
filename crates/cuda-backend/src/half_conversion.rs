use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

#[test]
#[ignore = "Nécessite RTX 4090 et CUDA 13.3"]
fn every_finite_half_scale_converts_exactly_in_q4_and_q6() {
    let source = format!(
        "{}\n{}\n{}",
        include_str!("../../../kernels/ada/q4_matvec.cu"),
        include_str!("../../../kernels/ada/embedding_q6.cu"),
        r#"extern "C" __global__ void convert_all(float* q4,float* q6){
            unsigned int h=blockIdx.x*blockDim.x+threadIdx.x;
            if(h<65536){q4[h]=decode_half(h);q6[h]=embedding_half(h);}
        }"#
    );
    let ptx = compile_ptx_with_opts(
        source,
        CompileOptions {
            arch: Some("compute_89"),
            ..Default::default()
        },
    )
    .unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let kernel = module.load_function("convert_all").unwrap();
    let stream = ctx.default_stream();
    let mut a = stream.alloc_zeros::<f32>(65536).unwrap();
    let mut b = stream.alloc_zeros::<f32>(65536).unwrap();
    let mut launch = stream.launch_builder(&kernel);
    launch.arg(&mut a).arg(&mut b);
    // SAFETY: 65536 bounded writes into two distinct buffers of that length.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .unwrap();
    let a = stream.clone_dtoh(&a).unwrap();
    let b = stream.clone_dtoh(&b).unwrap();
    stream.synchronize().unwrap();
    let mut checked = 0;
    for h in 0u32..65536 {
        let exponent = ((h >> 10) & 31) as i32;
        if exponent == 31 {
            continue;
        } // GGUF constructors reject non-finite scales.
        let fraction = f64::from(h & 1023);
        let magnitude = if exponent == 0 {
            fraction * 2f64.powi(-24)
        } else {
            (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15)
        };
        let expected = (if h & 32768 != 0 {
            -magnitude
        } else {
            magnitude
        }) as f32;
        assert_eq!(
            a[h as usize].to_bits(),
            expected.to_bits(),
            "Q4 half={h:04x}"
        );
        assert_eq!(
            b[h as usize].to_bits(),
            expected.to_bits(),
            "Q6 half={h:04x}"
        );
        checked += 1;
    }
    assert_eq!(checked, 63488);
}
