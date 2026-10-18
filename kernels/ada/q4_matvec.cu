// Première implémentation de référence GPU : un warp par ligne, quatre par bloc.
// Poids compressés conservés en Q4_0, aucune matrice FP32 intermédiaire.
// The loader rejects non-finite scales. Native conversion preserves all finite
// FP16 encodings exactly, including signed zero and subnormals.
__device__ __forceinline__ float decode_half(unsigned int h) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"((unsigned short)h));
    return value;
}
// Same arithmetic for isolated and fused projections.
__device__ float q4_row(const unsigned char* weights,const float* x,unsigned int columns,unsigned int row,unsigned int partition=0,unsigned int partitions=1){
    const unsigned int lane=threadIdx.x&31;
    const unsigned int blocks = columns / 32;
    const unsigned int j=(lane&7)*2;
    float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
    for(unsigned int b=partition*4+lane/8;b<blocks;b+=4*partitions){
        const unsigned char* p=weights+((unsigned long long)row*blocks+b)*18;
        // Q4_0 blocks are 18 bytes: these 16-bit reads are always aligned.
        const float d=decode_half(*(const unsigned short*)p);
        const unsigned int packed=*(const unsigned short*)(p+2+j);
        const unsigned int a=packed&255,c=packed>>8;
        s0+=(d*(float)((int)(a&15)-8))*x[b*32+j];
        s1+=(d*(float)((int)(c&15)-8))*x[b*32+j+1];
        s2+=(d*(float)((int)(a>>4)-8))*x[b*32+j+16];
        s3+=(d*(float)((int)(c>>4)-8))*x[b*32+j+17];
    }
    float sum=(s0+s1)+(s2+s3);
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    return sum;
}
extern "C" __global__ void q4_matvec(const unsigned char* weights,const float* x,float* y,unsigned int columns,unsigned int rows){
    const unsigned int row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows)return;
    const float value=q4_row(weights,x,columns,row);
    if((threadIdx.x&31)==0)y[row]=value;
}
extern "C" __global__ void q4_gate_up(const unsigned char* gate,const unsigned char* up,const float* x,float* g,float* u,unsigned int columns,unsigned int rows){
    unsigned int row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows*2)return;
    const bool first=row<rows;if(!first)row-=rows;
    const float value=q4_row(first?gate:up,x,columns,row);
    if((threadIdx.x&31)==0){if(first)g[row]=value;else u[row]=value;}
}
extern "C" __global__ void q4_qkv(const unsigned char* qw,const unsigned char* kw,const unsigned char* vw,const float* x,float* q,float* k,float* v,unsigned int columns,unsigned int qrows,unsigned int krows,unsigned int shared){
    unsigned int row=blockIdx.x*4+threadIdx.x/32;
    if(row>=qrows+krows+(shared?0:krows))return;
    const unsigned char* w=qw;float* output=q;
    if(row>=qrows){row-=qrows;w=kw;output=k;if(row>=krows){row-=krows;w=vw;output=v;}}
    const float value=q4_row(w,x,columns,row);
    if((threadIdx.x&31)==0)output[row]=value;
}

// Four warps share the input dimension of one row. Native FP32 activations.
extern "C" __global__ void q4_qkv_kparallel(const unsigned char* qw,const unsigned char* kw,const unsigned char* vw,const float* x,float* q,float* k,float* v,unsigned int columns,unsigned int qrows,unsigned int krows,unsigned int shared){
    unsigned int row=blockIdx.x;
    if(row>=qrows+krows+(shared?0:krows))return;
    const unsigned char* w=qw;float* output=q;
    if(row>=qrows){row-=qrows;w=kw;output=k;if(row>=krows){row-=krows;w=vw;output=v;}}
    __shared__ float partial[4];
    const unsigned int warp=threadIdx.x/32,lane=threadIdx.x&31;
    const float sum=q4_row(w,x,columns,row,warp,4);
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(threadIdx.x==0)output[row]=(partial[0]+partial[1])+(partial[2]+partial[3]);
}
extern "C" __global__ void q4_matvec_kparallel(const unsigned char* weights,
    const float* x,float* y,unsigned int columns,unsigned int rows) {
    const unsigned int row=blockIdx.x;
    if(row>=rows)return;
    __shared__ float partial[4];
    const unsigned int warp=threadIdx.x/32,lane=threadIdx.x&31;
    const float sum=q4_row(weights,x,columns,row,warp,4);
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(threadIdx.x==0)y[row]=(partial[0]+partial[1])+(partial[2]+partial[3]);
}
