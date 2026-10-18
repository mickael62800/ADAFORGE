// Un bloc de 256 threads par vecteur. Échelle directe : y = x * rsqrt(mean(x²)+eps) * w.
template<bool WithScale>
__device__ void rms_norm_impl(const float* x, const float* weights,
    float* y, unsigned int width, float epsilon)
{
    __shared__ float partial[32];
    const unsigned int thread=threadIdx.x;
    // Permute the original 256 logical lanes. Eight-lane shuffles perform
    // the original strides 128,64,32, retaining the FP32 reduction order.
    const unsigned int lane=(thread&7)*32+(thread>>3);
    const unsigned long long base=(unsigned long long)blockIdx.x*width;
    float sum=0.0f;
    // Fast path (width <= 4096): issue every load before the dependent sums.
    // Accumulation order per lane is unchanged, so results stay bit-identical.
    const bool cached=width<=4096;
    float v[16],w[16];
    if(cached){
        #pragma unroll
        for(unsigned int k=0;k<16;++k){const unsigned int i=lane+256*k;v[k]=i<width?x[base+i]:0.0f;
            w[k]=(WithScale&&i<width)?weights[i]:0.0f;}
        #pragma unroll
        for(unsigned int k=0;k<16;++k)if(lane+256*k<width)sum+=v[k]*v[k];
    }else{
        for(unsigned int i=lane;i<width;i+=256){float value=x[base+i];sum+=value*value;}
    }
    for(unsigned int offset=4;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset,8);
    if((thread&7)==0)partial[thread>>3]=sum;
    __syncthreads();
    if(thread<32){
        sum=partial[thread];
        for(unsigned int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(thread==0)partial[0]=sum;
    }
    __syncthreads();
    const float inverse_rms=rsqrtf(partial[0]/(float)width+epsilon);
    if(cached){
        #pragma unroll
        for(unsigned int k=0;k<16;++k){const unsigned int i=lane+256*k;
            if(i<width)y[base+i]=WithScale ? (v[k]*inverse_rms)*w[k] : v[k]*inverse_rms;}
        return;
    }
    for (unsigned int i=lane; i<width; i+=256)
        y[base+i]=WithScale ? (x[base+i]*inverse_rms)*weights[i] : x[base+i]*inverse_rms;
}

extern "C" __global__ void rms_norm(const float* x, const float* weights,
    float* y, unsigned int width, float epsilon)
{
    rms_norm_impl<true>(x,weights,y,width,epsilon);
}
// Gemma 4 V : normalisation par tête sans poids d’échelle.
extern "C" __global__ void rms_norm_unscaled(const float* x,
    float* y, unsigned int width, float epsilon)
{
    rms_norm_impl<false>(x,nullptr,y,width,epsilon);
}

// ---- Fused residual chains (single 256-thread block, width <= 4096) ----
// Each thread owns elements lane+256k (same permuted lanes as rms_norm_impl),
// so every intermediate stays in registers. Reductions and element formulas
// are copied verbatim from rms_norm_impl / vector_add / ffn_residual_scale:
// results are bit-identical to the unfused kernel sequence.
__device__ __forceinline__ float rms_inverse_cached(const float (&v)[16],unsigned int lane,unsigned int width,float epsilon){
    __shared__ float partial[32];
    const unsigned int thread=threadIdx.x;
    float sum=0.0f;
    #pragma unroll
    for(unsigned int k=0;k<16;++k)if(lane+256*k<width)sum+=v[k]*v[k];
    for(unsigned int offset=4;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset,8);
    __syncthreads(); // partial may still be read by a previous call
    if((thread&7)==0)partial[thread>>3]=sum;
    __syncthreads();
    if(thread<32){
        sum=partial[thread];
        for(unsigned int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(thread==0)partial[0]=sum;
    }
    __syncthreads();
    return rsqrtf(partial[0]/(float)width+epsilon);
}
// h = residual + rms(projected)*s1 ; normalized = rms(h)*s2
extern "C" __global__ void rms_add_rms(const float* projected,const float* residual,const float* s1,const float* s2,
    float* h_out,float* normalized,unsigned int width,float eps1,float eps2){
    // One block per vector (batched prefill); grid 1 is the single-token case.
    const unsigned long long base=(unsigned long long)blockIdx.x*width;
    projected+=base;residual+=base;h_out+=base;normalized+=base;
    const unsigned int thread=threadIdx.x,lane=(thread&7)*32+(thread>>3);
    // All loads issued up front: scales are usually cold (DRAM) by now.
    float v[16],r[16],a[16],b[16];
    #pragma unroll
    for(unsigned int k=0;k<16;++k){const unsigned int i=lane+256*k;const bool ok=i<width;
        v[k]=ok?projected[i]:0.0f;r[k]=ok?residual[i]:0.0f;a[k]=ok?s1[i]:0.0f;b[k]=ok?s2[i]:0.0f;}
    const float inv1=rms_inverse_cached(v,lane,width,eps1);
    #pragma unroll
    for(unsigned int k=0;k<16;++k){const unsigned int i=lane+256*k;
        if(i<width){const float n=__fmul_rn(v[k]*inv1,a[k]);v[k]=r[k]+n; /* no FMA: matches separate kernels */h_out[i]=v[k];}else v[k]=0.0f;}
    const float inv2=rms_inverse_cached(v,lane,width,eps2);
    #pragma unroll
    for(unsigned int k=0;k<16;++k){const unsigned int i=lane+256*k;if(i<width)normalized[i]=(v[k]*inv2)*b[k];}
}
// output = (rms(down)*s + residual) * scale
extern "C" __global__ void rms_residual_scale(const float* down,const float* residual,const float* s,
    float* output,unsigned int width,float eps,float scale){
    const unsigned long long base=(unsigned long long)blockIdx.x*width;
    down+=base;residual+=base;output+=base;
    const unsigned int thread=threadIdx.x,lane=(thread&7)*32+(thread>>3);
    float v[16],r[16],a[16];
    #pragma unroll
    for(unsigned int k=0;k<16;++k){const unsigned int i=lane+256*k;const bool ok=i<width;
        v[k]=ok?down[i]:0.0f;r[k]=ok?residual[i]:0.0f;a[k]=ok?s[i]:0.0f;}
    const float inv=rms_inverse_cached(v,lane,width,eps);
    #pragma unroll
    for(unsigned int k=0;k<16;++k){const unsigned int i=lane+256*k;
        if(i<width){const float post=__fmul_rn(v[k]*inv,a[k]);output[i]=(post+r[k])*scale; /* no FMA */}}
}
