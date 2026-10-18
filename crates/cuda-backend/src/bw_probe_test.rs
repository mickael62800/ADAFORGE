#[test]
#[ignore = "Requires RTX 4090 and CUDA"]
fn bandwidth_ceiling_probe() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg, sys};
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let src = r#"
extern "C" __global__ void readsum(const uint4* p,unsigned long long n,float* out){
    unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x,stride=(unsigned long long)gridDim.x*blockDim.x;
    unsigned int acc=0;
    for(;i<n;i+=stride){uint4 v=p[i];acc^=v.x^v.y^v.z^v.w;}
    if(acc==0x12345678u)out[0]=1.0f;
}"#;
    let m = ctx.load_module(cudarc::nvrtc::compile_ptx(src).unwrap()).unwrap();
    let f = m.load_function("readsum").unwrap();
    let bytes = 1usize << 30;
    let buf = stream.alloc_zeros::<u8>(bytes).unwrap();
    let mut out = stream.alloc_zeros::<f32>(1).unwrap();
    let n = (bytes / 16) as u64;
    for blocks in [128u32 * 4, 128 * 8, 128 * 16, 128 * 32] {
        let mut best = f32::MAX;
        for _ in 0..5 {
            let e0 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
            let mut l = stream.launch_builder(&f);
            l.arg(&buf).arg(&n).arg(&mut out);
            // SAFETY: grid-stride loop bounded by n uint4 elements of buf.
            unsafe { l.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
            let e1 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
            best = best.min(e0.elapsed_ms(&e1).unwrap());
        }
        println!("BW blocks {blocks}: {:.0} GB/s", bytes as f32 / best / 1e6);
    }
}
