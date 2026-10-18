template<int Unroll> __device__ void q4_gate_unrolled_body(const unsigned short* codes,
 const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows,float* up=nullptr){
    constexpr bool Split=true, Gate=false;
    const unsigned int lane=threadIdx.x&31;
    const unsigned int row=(Split?blockIdx.x:blockIdx.x*4+threadIdx.x/32)*4+(lane&3);
    if constexpr(!Split){if((row&~3u)>=rows)return;}
    const unsigned int j=(lane>>2)*2,blocks=columns==3840?120:columns/32;
    float a=0,b=0,c=0,d=0;
    #pragma unroll Unroll
    for(unsigned int k=Split?threadIdx.x/32:0;k<blocks;k+=Split?4:1){
        const unsigned long long tile=(unsigned long long)(row/4)*blocks+k;
        const float scale=decode_half(scales[tile*4+(lane&3)]);
        const unsigned int q=codes[(tile*8+(lane>>2))*4+(lane&3)];
        const unsigned int lo=q&255,hi=q>>8;
        a+=(scale*(float)((int)(lo&15)-8))*x[k*32+j];
        b+=(scale*(float)((int)(hi&15)-8))*x[k*32+j+1];
        c+=(scale*(float)((int)(lo>>4)-8))*x[k*32+j+16];
        d+=(scale*(float)((int)(hi>>4)-8))*x[k*32+j+17];
    }
    float sum=(a+b)+(c+d);
    if constexpr(Split){
        // Match native q4_row: reduce K partitions (0+2)+(1+3) FIRST,
        // then the eight channel lanes, retaining its exact FP32 tree.
        __shared__ float partial[128];
        partial[threadIdx.x]=sum;
        __syncthreads();
        if(threadIdx.x<32){
            sum=(partial[lane]+partial[64+lane])+(partial[32+lane]+partial[96+lane]);
            for(int offset=16;offset>=4;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
            if(lane<4 && row<rows){
                float* output=Gate && row>=rows/2?up:y;
                const unsigned int target=Gate && row>=rows/2?row-rows/2:row;
                output[target]=sum;
            }
        }
    }else{
        for(int offset=16;offset>=4;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(lane<4 && row<rows)y[row]=sum;
    }
}
extern "C" __global__ void q4_gate_unroll1(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){q4_gate_unrolled_body<1>(codes,scales,x,y,columns,rows);}
extern "C" __global__ void q4_gate_unroll2(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){q4_gate_unrolled_body<2>(codes,scales,x,y,columns,rows);}
extern "C" __global__ void q4_gate_unroll4(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){q4_gate_unrolled_body<4>(codes,scales,x,y,columns,rows);}
extern "C" __global__ void q4_gate_unroll8(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){q4_gate_unrolled_body<8>(codes,scales,x,y,columns,rows);}
