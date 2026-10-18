__device__ __forceinline__ unsigned short tiled_weight_load(const unsigned short* p){
#ifdef ADAFORGE_STREAM_WEIGHTS
 unsigned short v; asm("ld.global.cs.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v;
#else
 return *p;
#endif
}
// Research layout: tiles of four rows, block-major, words interleaved by row.
// Eight lanes per row; four rows per warp. Group size stays 32.
template<bool Split,bool Gate=false> __device__ void q4_tiled_body(const unsigned short* codes,
 const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows,float* up=nullptr){
    const unsigned int lane=threadIdx.x&31;
    const unsigned int row=(Split?blockIdx.x:blockIdx.x*4+threadIdx.x/32)*4+(lane&3);
    if constexpr(!Split){if((row&~3u)>=rows)return;}
    const unsigned int j=(lane>>2)*2,blocks=columns/32;
    float a=0,b=0,c=0,d=0;
    for(unsigned int k=Split?threadIdx.x/32:0;k<blocks;k+=Split?4:1){
        const unsigned long long tile=(unsigned long long)(row/4)*blocks+k;
        const float scale=decode_half(tiled_weight_load(scales+tile*4+(lane&3)));
        const unsigned int q=tiled_weight_load(codes+(tile*8+(lane>>2))*4+(lane&3));
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
extern "C" __global__ void q4_tiled(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){q4_tiled_body<false>(codes,scales,x,y,columns,rows);}
extern "C" __global__ void q4_tiled_split(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned int columns,unsigned int rows){q4_tiled_body<true>(codes,scales,x,y,columns,rows);}
extern "C" __global__ void q4_tiled_gate(const unsigned short* codes,const unsigned short* scales,const float* x,float* g,float* u,unsigned int columns,unsigned int rows){q4_tiled_body<true,true>(codes,scales,x,g,columns,rows*2,u);}
extern "C" __global__ void probe_evict(unsigned int* buffer,unsigned int count){
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)buffer[i]=buffer[i]*1664525u+i+1013904223u;
}

// Two 128-thread groups compute matching gate/up tiles. FP32 projections
// meet in shared memory before the unchanged GELU expression.
extern "C" __global__ void q4_tiled_gate_gelu(const unsigned short* codes,const unsigned short* scales,const float* x,float* output,unsigned columns,unsigned rows){
 const unsigned local=threadIdx.x%128, group=threadIdx.x/128,lane=local%32;
 const unsigned target=blockIdx.x*4+(lane&3), row=target+group*rows;
 const unsigned blocks=columns/32,j=(lane>>2)*2;
 float a=0,b=0,c=0,d=0;
 for(unsigned k=local/32;k<blocks;k+=4){
  const unsigned long long tile=(unsigned long long)(row/4)*blocks+k;
  const float scale=decode_half(tiled_weight_load(scales+tile*4+(lane&3)));
  const unsigned q=tiled_weight_load(codes+(tile*8+(lane>>2))*4+(lane&3)),lo=q&255,hi=q>>8;
  a+=(scale*(float)((int)(lo&15)-8))*x[k*32+j];
  b+=(scale*(float)((int)(hi&15)-8))*x[k*32+j+1];
  c+=(scale*(float)((int)(lo>>4)-8))*x[k*32+j+16];
  d+=(scale*(float)((int)(hi>>4)-8))*x[k*32+j+17];
 }
 __shared__ float partial[256]; __shared__ float projected[8];
 partial[threadIdx.x]=(a+b)+(c+d); __syncthreads();
 if(local<32){const float* p=partial+group*128;
 float sum=(p[lane]+p[64+lane])+(p[32+lane]+p[96+lane]);
 for(int offset=16;offset>=4;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
 if(lane<4)projected[group*4+lane]=sum;}
 __syncthreads();
 if(threadIdx.x<4 && target<rows){const float g=projected[threadIdx.x];
 const float activation=0.5f*g*(1.0f+tanhf(0.7978845608028654f*(g+0.044715f*g*g*g)));
 output[target]=activation*projected[4+threadIdx.x];}
}
