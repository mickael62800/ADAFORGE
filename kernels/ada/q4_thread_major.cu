// Thread-major Q4_0 layout: same arithmetic and FP32 reduction tree as q4_row,
// but each thread reads its next eight 16-bit code words (and its group's eight
// scales) with one 16-byte load. Layout is produced by matvec::thread_major.
// Per (row, partition): C full chunks of uint4 [chunk][lane], then a tail of
// R u16 words [i][lane]; scales use the same shape with 4 groups instead of 32 lanes.
__device__ __forceinline__ void q4_tm_term(unsigned int packed,float d,const float* x,unsigned int b,unsigned int j,
    float& s0,float& s1,float& s2,float& s3){
    const unsigned int a=packed&255,c=packed>>8;
    s0+=(d*(float)((int)(a&15)-8))*x[b*32+j];
    s1+=(d*(float)((int)(c&15)-8))*x[b*32+j+1];
    s2+=(d*(float)((int)(a>>4)-8))*x[b*32+j+16];
    s3+=(d*(float)((int)(c>>4)-8))*x[b*32+j+17];
}
__device__ float q4_tm_row(const unsigned char* codes,const unsigned char* scales,const float* x,
    unsigned int columns,unsigned int row,unsigned int partition,unsigned int partitions){
    const unsigned int lane=threadIdx.x&31,group=lane/8,j=(lane&7)*2;
    const unsigned int blocks=columns/32,stride=4*partitions;
    const unsigned int iterations=(blocks+stride-1)/stride,chunks=iterations/8,tail=iterations%8;
    const unsigned long long slot=(unsigned long long)row*partitions+partition;
    const unsigned long long code_bytes=(unsigned long long)chunks*512+tail*64;
    const unsigned long long scale_bytes=(unsigned long long)chunks*64+tail*8;
    const uint4* cv=(const uint4*)(codes+slot*code_bytes);
    const uint4* sv=(const uint4*)(scales+slot*scale_bytes);
    float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
    unsigned int b=partition*4+group;
    // Software pipeline: the next chunk is in flight while this one is consumed.
    uint4 q=make_uint4(0,0,0,0),s=make_uint4(0,0,0,0);
    if(chunks){q=cv[lane];s=sv[group];}
    for(unsigned int c=0;c<chunks;++c){
        uint4 nq=q,ns=s;
        if(c+1<chunks){nq=cv[(c+1)*32+lane];ns=sv[(c+1)*4+group];}
        const unsigned int qw[4]={q.x,q.y,q.z,q.w},sw[4]={s.x,s.y,s.z,s.w};
        #pragma unroll
        for(unsigned int i=0;i<8;++i){
            if(b<blocks){
                const unsigned int word=(qw[i/2]>>((i&1)*16))&65535u;
                const float d=decode_half((sw[i/2]>>((i&1)*16))&65535u);
                q4_tm_term(word,d,x,b,j,s0,s1,s2,s3);
            }
            b+=stride;
        }
        q=nq;s=ns;
    }
    const unsigned short* ct=(const unsigned short*)(codes+slot*code_bytes+(unsigned long long)chunks*512);
    const unsigned short* st=(const unsigned short*)(scales+slot*scale_bytes+(unsigned long long)chunks*64);
    for(unsigned int i=0;i<tail;++i){
        if(b<blocks)q4_tm_term(ct[i*32+lane],decode_half(st[i*4+group]),x,b,j,s0,s1,s2,s3);
        b+=stride;
    }
    float sum=(s0+s1)+(s2+s3);
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    return sum;
}
extern "C" __global__ void q4_tm_matvec(const unsigned char* codes,const unsigned char* scales,
    const float* x,float* y,unsigned int columns,unsigned int rows){
    const unsigned int row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows)return;
    const float value=q4_tm_row(codes,scales,x,columns,row,0,1);
    if((threadIdx.x&31)==0)y[row]=value;
}
extern "C" __global__ void q4_tm_matvec_kparallel(const unsigned char* codes,const unsigned char* scales,
    const float* x,float* y,unsigned int columns,unsigned int rows){
    const unsigned int row=blockIdx.x;
    if(row>=rows)return;
    __shared__ float partial[4];
    const unsigned int warp=threadIdx.x/32,lane=threadIdx.x&31;
    const float sum=q4_tm_row(codes,scales,x,columns,row,warp,4);
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(threadIdx.x==0)y[row]=(partial[0]+partial[1])+(partial[2]+partial[3]);
}
extern "C" __global__ void q4_tm_qkv_kparallel(const unsigned char* qc,const unsigned char* qs,
    const unsigned char* kc,const unsigned char* ks,const unsigned char* vc,const unsigned char* vs,
    const float* x,float* q,float* k,float* v,unsigned int columns,unsigned int qrows,unsigned int krows,unsigned int shared){
    unsigned int row=blockIdx.x;
    if(row>=qrows+krows+(shared?0:krows))return;
    const unsigned char* c=qc;const unsigned char* s=qs;float* output=q;
    if(row>=qrows){row-=qrows;c=kc;s=ks;output=k;if(row>=krows){row-=krows;c=vc;s=vs;output=v;}}
    __shared__ float partial[4];
    const unsigned int warp=threadIdx.x/32,lane=threadIdx.x&31;
    const float sum=q4_tm_row(c,s,x,columns,row,warp,4);
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(threadIdx.x==0)output[row]=(partial[0]+partial[1])+(partial[2]+partial[3]);
}

// Native GGUF layout, loads of U iterations hoisted before their arithmetic.
// Per-accumulator order is unchanged, so results are bit-identical to q4_row.
template<int U> __device__ float q4_row_hoisted(const unsigned char* weights,const float* x,unsigned int columns,
    unsigned int row,unsigned int partition,unsigned int partitions){
    const unsigned int lane=threadIdx.x&31,blocks=columns/32,j=(lane&7)*2,stride=4*partitions;
    float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
    const unsigned char* base=weights+(unsigned long long)row*blocks*18;
    unsigned int b=partition*4+lane/8;
    for(;b+(U-1)*stride<blocks;b+=U*stride){
        unsigned int dh[U],pk[U];
        #pragma unroll
        for(int u=0;u<U;++u){const unsigned char* p=base+(b+u*stride)*18;dh[u]=*(const unsigned short*)p;pk[u]=*(const unsigned short*)(p+2+j);}
        #pragma unroll
        for(int u=0;u<U;++u)q4_tm_term(pk[u],decode_half(dh[u]),x,b+u*stride,j,s0,s1,s2,s3);
    }
    for(;b<blocks;b+=stride){const unsigned char* p=base+b*18;
        q4_tm_term(*(const unsigned short*)(p+2+j),decode_half(*(const unsigned short*)p),x,b,j,s0,s1,s2,s3);}
    float sum=(s0+s1)+(s2+s3);
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    return sum;
}
extern "C" __global__ void q4_h_matvec_kparallel(const unsigned char* weights,const float* x,float* y,unsigned int columns,unsigned int rows){
    const unsigned int row=blockIdx.x;
    if(row>=rows)return;
    __shared__ float partial[4];
    const unsigned int warp=threadIdx.x/32,lane=threadIdx.x&31;
    const float sum=q4_row_hoisted<4>(weights,x,columns,row,warp,4);
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(threadIdx.x==0)y[row]=(partial[0]+partial[1])+(partial[2]+partial[3]);
}
extern "C" __global__ void q4_h_matvec(const unsigned char* weights,const float* x,float* y,unsigned int columns,unsigned int rows){
    const unsigned int row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows)return;
    const float value=q4_row_hoisted<4>(weights,x,columns,row,0,1);
    if((threadIdx.x&31)==0)y[row]=value;
}
