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
    float v[16];
    if(cached){
        #pragma unroll
        for(unsigned int k=0;k<16;++k){const unsigned int i=lane+256*k;v[k]=i<width?x[base+i]:0.0f;}
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
            if(i<width)y[base+i]=WithScale ? (v[k]*inverse_rms)*weights[i] : v[k]*inverse_rms;}
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
