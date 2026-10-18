__device__ __forceinline__ void tiled_copy(void* dst,const void* src){
    const unsigned int address=__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(address),"l"(src):"memory");
}
__device__ __forceinline__ void tiled_scale_copy(void* dst,const void* src){
    const unsigned int address=__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"::"r"(address),"l"(src):"memory");
}
template<bool Gate,int Groups=1> __device__ void q4_tiled_async_body(const unsigned short* codes,
 const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows,float* up=nullptr){
    __shared__ __align__(16) unsigned short qbuf[4][2][Groups][32];
    __shared__ __align__(16) unsigned short sbuf[4][2][Groups][4];
    __shared__ float partial[128];
    const unsigned int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const unsigned int row=blockIdx.x*4+(lane&3),j=(lane>>2)*2,blocks=columns/32;
    const auto issue=[&](unsigned int k,unsigned int stage){
        if(lane<Groups*4 && k+(lane/4)*4<blocks){
            const unsigned int group=lane/4,vec=lane&3;
            const unsigned long long tile=(unsigned long long)blockIdx.x*blocks+k+group*4;
            tiled_copy(qbuf[warp][stage][group]+vec*8,codes+tile*32+vec*8);
        }
        if(lane<Groups*2 && k+(lane/2)*4<blocks){
            const unsigned int group=lane/2,vec=lane&1;
            const unsigned long long tile=(unsigned long long)blockIdx.x*blocks+k+group*4;
            tiled_scale_copy(sbuf[warp][stage][group]+vec*2,scales+tile*4+vec*2);
        }
        asm volatile("cp.async.commit_group;":::"memory");
    };
    issue(warp,0);
    float a=0,b=0,c=0,d=0;
    for(unsigned int step=0,k0=warp;step<(blocks+Groups*4-1)/(Groups*4);++step,k0+=Groups*4){
        const unsigned int stage=step&1;
        if(k0+Groups*4<blocks){
            issue(k0+Groups*4,stage^1);
            asm volatile("cp.async.wait_group 1;":::"memory");
        }else{asm volatile("cp.async.wait_group 0;":::"memory");}
        __syncwarp();
        #pragma unroll
        for(unsigned int group=0;group<Groups;++group){
          const unsigned int k=k0+group*4;
          if(k<blocks){
            const float scale=decode_half(sbuf[warp][stage][group][lane&3]);
            const unsigned int q=qbuf[warp][stage][group][lane],lo=q&255,hi=q>>8;
            a+=(scale*(float)((int)(lo&15)-8))*x[k*32+j];
            b+=(scale*(float)((int)(hi&15)-8))*x[k*32+j+1];
            c+=(scale*(float)((int)(lo>>4)-8))*x[k*32+j+16];
            d+=(scale*(float)((int)(hi>>4)-8))*x[k*32+j+17];
          }
        }
        __syncwarp();
    }
    partial[threadIdx.x]=(a+b)+(c+d);
    __syncthreads();
    if(threadIdx.x<32){
        float sum=(partial[lane]+partial[64+lane])+(partial[32+lane]+partial[96+lane]);
        for(int offset=16;offset>=4;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(lane<4 && row<rows){
            float* output=Gate && row>=rows/2?up:y;
            output[Gate && row>=rows/2?row-rows/2:row]=sum;
        }
    }
}
extern "C" __global__ void q4_tiled_async(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){q4_tiled_async_body<false>(codes,scales,x,y,columns,rows);}
extern "C" __global__ void q4_tiled_async4(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){q4_tiled_async_body<false,4>(codes,scales,x,y,columns,rows);}
extern "C" __global__ void q4_tiled_gate_async(const unsigned short* codes,const unsigned short* scales,const float* x,float* g,float* u,unsigned int columns,unsigned int rows){q4_tiled_async_body<true>(codes,scales,x,g,columns,rows*2,u);}
