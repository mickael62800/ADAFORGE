// Experimental Q8 activations: FP32 scale per 32 values, signed INT8 [-127,127].
extern "C" __global__ void quantize_q8(const float* x, signed char* q, float* scales, unsigned int columns) {
    const unsigned int lane=threadIdx.x&31;
    const unsigned int block=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(block>=columns/32)return;
    const float value=x[block*32+lane];
    float maximum=fabsf(value);
    for(int mask=16;mask>0;mask>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,mask));
    const bool invalid=__any_sync(0xffffffff,!isfinite(value));
    const float d=invalid?__int_as_float(0x7fc00000):maximum/127.0f;
    int v=(d==0.0f||invalid)?0:__float2int_rn(value/d);
    v=max(-127,min(127,v));
    q[block*32+lane]=(signed char)v;
    if(lane==0)scales[block]=d;
}
__device__ int dot4(int w,int x,int sum) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(w),"r"(x),"r"(sum));
    return result;
}
__device__ float q4_q8_fragment(const unsigned char* weights,const signed char* x,const float* scales,
    unsigned int columns,unsigned int row,unsigned int start,unsigned int stride) {
    const unsigned int blocks=columns/32;
    float sum=0;
    for(unsigned int b=start;b<blocks;b+=stride) {
        const unsigned char* w=weights+((unsigned long long)row*blocks+b)*18;
        const int* xp=(const int*)(x+b*32);
        int dot=0;
        #pragma unroll
        for(unsigned int j=0;j<4;++j) {
            // Native 18-byte blocks only guarantee 16-bit alignment.
            const unsigned int packed=*(const unsigned short*)(w+2+j*4)
                |((unsigned int)*(const unsigned short*)(w+4+j*4)<<16);
            const unsigned int low=packed&0x0f0f0f0f,high=(packed>>4)&0x0f0f0f0f;
            const int lo=(int)(((low|0x80808080u)-0x08080808u)^0x80808080u);
            const int hi=(int)(((high|0x80808080u)-0x08080808u)^0x80808080u);
            dot=dot4(lo,xp[j],dot);
            dot=dot4(hi,xp[j+4],dot);
        }
        const float d=decode_half(*(const unsigned short*)w);
        sum+=(d*scales[b])*(float)dot;
    }
    return sum;
}
extern "C" __global__ void q4_q8_matvec(const unsigned char* weights,const signed char* x,const float* scales,
    float* output,unsigned int columns,unsigned int rows) {
    const unsigned int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;
    if(row>=rows)return;
    float sum=q4_q8_fragment(weights,x,scales,columns,row,lane,32);
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0)output[row]=sum;
}
extern "C" __global__ void q4_q8_kparallel(const unsigned char* weights,const signed char* x,const float* scales,
    float* output,unsigned int columns,unsigned int rows) {
    const unsigned int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x/32;
    if(row>=rows)return;
    __shared__ float partial[4];
    float sum=q4_q8_fragment(weights,x,scales,columns,row,threadIdx.x,128);
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(threadIdx.x==0)output[row]=(partial[0]+partial[1])+(partial[2]+partial[3]);
}

extern "C" __global__ void q4_q8_gate_up(const unsigned char* gate,const unsigned char* up,
    const signed char* x,const float* scales,float* g,float* u,unsigned int columns,unsigned int rows) {
    unsigned int row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows*2)return;
    const bool first=row<rows;if(!first)row-=rows;
    float sum=q4_q8_fragment(first?gate:up,x,scales,columns,row,threadIdx.x&31,32);
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if((threadIdx.x&31)==0){if(first)g[row]=sum;else u[row]=sum;}
}
