// Batched Q4_0 matvec: weights are read once for up to Q4_MAXB tokens.
// For every token the arithmetic, accumulation order and reduction tree are
// exactly those of q4_row / q4_tiled_body, so each output is bit-identical to
// the single-token kernels. x is [n][columns], y is [n][rows].
#ifndef Q4_MAXB
#define Q4_MAXB 8
#endif
template<int B> __device__ __forceinline__ void q4_batch_row(const unsigned char* weights,const float* x,unsigned int columns,
    unsigned int row,unsigned int partition,unsigned int partitions,float (&out)[B]){
    const unsigned int lane=threadIdx.x&31,blocks=columns/32,j=(lane&7)*2;
    float s0[B],s1[B],s2[B],s3[B];
    #pragma unroll
    for(int t=0;t<B;++t){s0[t]=0.0f;s1[t]=0.0f;s2[t]=0.0f;s3[t]=0.0f;}
    // Next iteration's weights are loaded while this one is computed.
    const unsigned int step=4*partitions;
    unsigned int b=partition*4+lane/8;
    const unsigned char* rowp=weights+(unsigned long long)row*blocks*18;
    unsigned int nd=0,np=0;
    if(b<blocks){nd=*(const unsigned short*)(rowp+b*18);np=*(const unsigned short*)(rowp+b*18+2+j);}
    for(;b<blocks;b+=step){
        const unsigned int dh=nd,packed=np;
        if(b+step<blocks){nd=*(const unsigned short*)(rowp+(b+step)*18);np=*(const unsigned short*)(rowp+(b+step)*18+2+j);}
        const float d=decode_half(dh);
        const unsigned int a=packed&255,c=packed>>8;
        const float w0=d*(float)((int)(a&15)-8),w1=d*(float)((int)(c&15)-8);
        const float w2=d*(float)((int)(a>>4)-8),w3=d*(float)((int)(c>>4)-8);
        #pragma unroll
        for(int t=0;t<B;++t){
            // x is pre-permuted (q4_permute_x): [j, j+1, j+16, j+17] are contiguous.
            const float4 xv=*(const float4*)(x+(unsigned long long)t*columns+b*32+j*2);
            s0[t]+=w0*xv.x;s1[t]+=w1*xv.y;s2[t]+=w2*xv.z;s3[t]+=w3*xv.w;
        }
    }
    #pragma unroll
    for(int t=0;t<B;++t){
        float sum=(s0[t]+s1[t])+(s2[t]+s3[t]);
        for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        out[t]=sum;
    }
}
template<int B> __device__ void q4_batch_matvec_t(const unsigned char* weights,const float* x,float* y,
    unsigned int columns,unsigned int rows){
    const unsigned int row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows)return;
    float out[B];
    q4_batch_row<B>(weights,x,columns,row,0,1,out);
    if((threadIdx.x&31)==0){
        #pragma unroll
        for(int t=0;t<B;++t)y[(unsigned long long)t*rows+row]=out[t];
    }
}
template<int B> __device__ void q4_batch_kparallel_t(const unsigned char* weights,const float* x,float* y,
    unsigned int columns,unsigned int rows){
    const unsigned int row=blockIdx.x;
    if(row>=rows)return;
    __shared__ float partial[4][B];
    const unsigned int warp=threadIdx.x/32,lane=threadIdx.x&31;
    float out[B];
    q4_batch_row<B>(weights,x,columns,row,warp,4,out);
    if(lane==0){
        #pragma unroll
        for(int t=0;t<B;++t)partial[warp][t]=out[t];
    }
    __syncthreads();
    if(threadIdx.x<B)y[(unsigned long long)threadIdx.x*rows+row]=
        (partial[0][threadIdx.x]+partial[1][threadIdx.x])+(partial[2][threadIdx.x]+partial[3][threadIdx.x]);
}
#define Q4_BATCH_INST(B) extern "C" __global__ void q4_batch_matvec_b##B(const unsigned char* w,const float* x,float* y,unsigned int c,unsigned int r){q4_batch_matvec_t<B>(w,x,y,c,r);} extern "C" __global__ void q4_batch_kparallel_b##B(const unsigned char* w,const float* x,float* y,unsigned int c,unsigned int r){q4_batch_kparallel_t<B>(w,x,y,c,r);}
Q4_BATCH_INST(1) Q4_BATCH_INST(2) Q4_BATCH_INST(3) Q4_BATCH_INST(4)
Q4_BATCH_INST(5) Q4_BATCH_INST(6) Q4_BATCH_INST(7) Q4_BATCH_INST(8)
Q4_BATCH_INST(9) Q4_BATCH_INST(10) Q4_BATCH_INST(11) Q4_BATCH_INST(12)
Q4_BATCH_INST(13) Q4_BATCH_INST(14) Q4_BATCH_INST(15) Q4_BATCH_INST(16)

// Within every 32-value group, element p*4+{0,1,2,3} <- {2p, 2p+1, 2p+16, 2p+17}.
extern "C" __global__ void q4_permute_x(const float* x,float* xp,unsigned int total){
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total)return;
    const unsigned int base=i&~31u,o=i&31u,p=o/4,k=o%4;
    xp[i]=x[base+2*p+(k&1)+(k>>1)*16];
}

// Batched q4_tiled_gate (Split + Gate): same per-token arithmetic and the
// same (0+2)+(1+3) cross-warp tree as q4_tiled_body<true,true>.
// x permuted (q4_permute_x), [B][columns]; g/u are [B][rows/2].
template<int B> __device__ void q4_batch_tiled_gate_t(const unsigned short* codes,const unsigned short* scales,
    const float* x,float* g,float* u,unsigned int columns,unsigned int rows){
    const unsigned int lane=threadIdx.x&31;
    const unsigned int row=blockIdx.x*4+(lane&3);
    const unsigned int j=(lane>>2)*2,blocks=columns/32;
    float a[B],b[B],c[B],d[B];
    #pragma unroll
    for(int t=0;t<B;++t){a[t]=0.0f;b[t]=0.0f;c[t]=0.0f;d[t]=0.0f;}
    const unsigned long long tile0=(unsigned long long)(row/4)*blocks;
    unsigned int k=threadIdx.x/32,ns=0,nq=0;
    if(k<blocks){ns=scales[(tile0+k)*4+(lane&3)];nq=codes[((tile0+k)*8+(lane>>2))*4+(lane&3)];}
    for(;k<blocks;k+=4){
        const unsigned int sh=ns,q=nq;
        if(k+4<blocks){ns=scales[(tile0+k+4)*4+(lane&3)];nq=codes[((tile0+k+4)*8+(lane>>2))*4+(lane&3)];}
        const float scale=decode_half(sh);
        const unsigned int lo=q&255,hi=q>>8;
        const float w0=scale*(float)((int)(lo&15)-8),w1=scale*(float)((int)(hi&15)-8);
        const float w2=scale*(float)((int)(lo>>4)-8),w3=scale*(float)((int)(hi>>4)-8);
        #pragma unroll
        for(int t=0;t<B;++t){
            const float4 xv=*(const float4*)(x+(unsigned long long)t*columns+k*32+j*2);
            a[t]+=w0*xv.x;b[t]+=w1*xv.y;c[t]+=w2*xv.z;d[t]+=w3*xv.w;
        }
    }
    __shared__ float partial[B][128];
    #pragma unroll
    for(int t=0;t<B;++t)partial[t][threadIdx.x]=(a[t]+b[t])+(c[t]+d[t]);
    __syncthreads();
    if(threadIdx.x<32){
        const unsigned int half=rows/2;
        #pragma unroll
        for(int t=0;t<B;++t){
            float sum=(partial[t][lane]+partial[t][64+lane])+(partial[t][32+lane]+partial[t][96+lane]);
            for(int offset=16;offset>=4;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
            if(lane<4 && row<rows){
                if(row>=half)u[(unsigned long long)t*half+row-half]=sum;
                else g[(unsigned long long)t*half+row]=sum;
            }
        }
    }
}
#define Q4_TILED_BATCH_INST(B) \
extern "C" __global__ void q4_batch_tiled_gate_b##B(const unsigned short* c,const unsigned short* s,const float* x,float* g,float* u,unsigned int columns,unsigned int rows){q4_batch_tiled_gate_t<B>(c,s,x,g,u,columns,rows*2);}
Q4_TILED_BATCH_INST(1) Q4_TILED_BATCH_INST(2) Q4_TILED_BATCH_INST(3) Q4_TILED_BATCH_INST(4)
Q4_TILED_BATCH_INST(5) Q4_TILED_BATCH_INST(6) Q4_TILED_BATCH_INST(7) Q4_TILED_BATCH_INST(8)
Q4_TILED_BATCH_INST(9) Q4_TILED_BATCH_INST(10) Q4_TILED_BATCH_INST(11) Q4_TILED_BATCH_INST(12)
Q4_TILED_BATCH_INST(13) Q4_TILED_BATCH_INST(14) Q4_TILED_BATCH_INST(15) Q4_TILED_BATCH_INST(16)
