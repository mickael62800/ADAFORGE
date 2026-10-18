extern "C" __global__ void gelu_gate(const float* gate,const float* up,float* output,unsigned int n)
{
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) {
        const float g=gate[i];
        const float activation=0.5f*g*(1.0f+tanhf(0.7978845608028654f*(g+0.044715f*g*g*g)));
        output[i]=activation*up[i];
    }
}
extern "C" __global__ void ffn_residual_scale(const float* normalized,const float* residual,
    float* output,unsigned int n,float scale)
{
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) output[i]=(normalized[i]+residual[i])*scale;
}
