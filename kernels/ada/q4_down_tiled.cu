// Four output rows per CTA, 16 K partitions; exact native down reduction.
template<unsigned int FixedBlocks,int Unroll>
__device__ void down_tiled_body(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){
    const unsigned int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const unsigned int r=lane&3,row=blockIdx.x*4+r,j=(lane>>2)*2;
    const unsigned int blocks=FixedBlocks?FixedBlocks:columns/32;
    float a=0,b=0,c=0,d=0;
    #pragma unroll Unroll
    for(unsigned int k=warp;k<blocks;k+=16){
        const unsigned long long tile=(unsigned long long)blockIdx.x*blocks+k;
        const float scale=decode_half(scales[tile*4+r]);
        const unsigned int packed=codes[(tile*8+(lane>>2))*4+r];
        const unsigned int lo=packed&255,hi=packed>>8;
        a+=(scale*(float)((int)(lo&15)-8))*x[k*32+j];
        b+=(scale*(float)((int)(hi&15)-8))*x[k*32+j+1];
        c+=(scale*(float)((int)(lo>>4)-8))*x[k*32+j+16];
        d+=(scale*(float)((int)(hi>>4)-8))*x[k*32+j+17];
    }
    __shared__ float partial[512];
    __shared__ float reduced[16];
    partial[threadIdx.x]=(a+b)+(c+d);
    __syncthreads();
    if(threadIdx.x<128){
        const unsigned int base=warp*128+lane;
        float sum=(partial[base]+partial[base+64])+(partial[base+32]+partial[base+96]);
        for(int offset=16;offset>=4;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(lane<4)reduced[warp*4+lane]=sum;
    }
    __syncthreads();
    if(threadIdx.x<4 && row<rows)y[row]=(reduced[r]+reduced[4+r])+(reduced[8+r]+reduced[12+r]);
}
#define DOWN_KERNEL(U) extern "C" __global__ void q4_down_tiled##U(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){if(columns==15360)down_tiled_body<480,U>(codes,scales,x,y,columns,rows);else down_tiled_body<0,U>(codes,scales,x,y,columns,rows);}
DOWN_KERNEL(1)
DOWN_KERNEL(2)
DOWN_KERNEL(4)
DOWN_KERNEL(8)
