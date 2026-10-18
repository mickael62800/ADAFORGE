// Un bloc de 256 threads par vecteur. Échelle directe : y = x * rsqrt(mean(x²)+eps) * w.
template<bool WithScale>
__device__ void rms_norm_impl(const float* x, const float* weights,
    float* y, unsigned int width, float epsilon)
{
    __shared__ float partial[256];
    const unsigned int lane = threadIdx.x;
    const unsigned long long base = (unsigned long long)blockIdx.x * width;
    float sum = 0.0f;
    for (unsigned int i=lane; i<width; i+=256) {
        const float v=x[base+i];
        sum += v*v;
    }
    partial[lane]=sum;
    __syncthreads();
    for (unsigned int stride=128; stride>0; stride>>=1) {
        if (lane<stride) partial[lane]+=partial[lane+stride];
        __syncthreads();
    }
    const float inverse_rms=rsqrtf(partial[0]/(float)width+epsilon);
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
