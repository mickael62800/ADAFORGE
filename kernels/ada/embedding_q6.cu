// Lignes Q6_K sélectionnées et compactées par le lecteur ; un thread par canal.
// The loader rejects non-finite scales. Native conversion preserves all finite
// FP16 encodings exactly, including signed zero and subnormals.
__device__ __forceinline__ float embedding_half(unsigned int h) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"((unsigned short)h));
    return value;
}
extern "C" __global__ void embedding_q6(const unsigned char* packed,float* output,unsigned int elements,float multiplier)
{
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=elements) return;
    const unsigned char* b=packed+(unsigned long long)(i/256)*210;
    const unsigned int within=i%256,group=within/128,quarter=(within%128)/32,lane=within%32;
    const unsigned int low_byte=b[group*64+(quarter%2)*32+lane];
    const unsigned int low=(quarter<2)?(low_byte&15):(low_byte>>4);
    const unsigned int high=(b[128+group*32+lane]>>(quarter*2))&3;
    const int quant=(int)(low|(high<<4))-32;
    const signed char subscale=(signed char)b[192+group*8+quarter*2+lane/16];
    const float d=embedding_half((unsigned int)b[208]|((unsigned int)b[209]<<8));
    output[i]=((d*(float)subscale)*(float)quant)*multiplier;
}
