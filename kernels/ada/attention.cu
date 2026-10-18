// Premier decode attention : un bloc par tête Q, cache [token][tête KV][canal].
template<bool ParallelExp> __device__ __forceinline__ void attention_impl(const float* q,const float* keys,const float* values,
    float* output,unsigned int width,unsigned int q_heads,unsigned int kv_heads,
    unsigned int tokens,unsigned int position,unsigned int window,float scale)
{
    extern __shared__ float scores[];
    const unsigned int thread=threadIdx.x,head=blockIdx.x;
    const unsigned int kv=head/(q_heads/kv_heads);
    const unsigned int end=min(tokens,position+1);
    const unsigned int start=(window>0 && end>window) ? end-window : 0;
    const unsigned long long stride=(unsigned long long)kv_heads*width;
    // Eight warps compute eight token scores concurrently; only the final
    // softmax and value accumulation require block-wide synchronization.
    const unsigned int warp=thread/32,lane=thread&31;
    if(width<=512){
        // q cached in registers; two tokens per warp iteration with all key
        // loads issued first. Per-token arithmetic and tree are unchanged.
        float qr[16];
        #pragma unroll
        for(unsigned int k=0;k<16;++k){const unsigned int c=lane+k*32;qr[k]=c<width?q[(unsigned long long)head*width+c]:0.0f;}
        const float* kb=keys+(unsigned long long)kv*width;
        for(unsigned int t=start+warp;t<end;t+=16){
            const bool second=t+8<end;
            float ka[16],kc[16];
            #pragma unroll
            for(unsigned int k=0;k<16;++k){const unsigned int c=lane+k*32;
                ka[k]=c<width?kb[t*stride+c]:0.0f;kc[k]=(second&&c<width)?kb[(t+8)*stride+c]:0.0f;}
            float pa[8]={0.0f},pc[8]={0.0f};
            // Element k belongs to group k%8 (c=lane+group*32+256*(k/8)), visited in increasing c.
            #pragma unroll
            for(unsigned int k=0;k<16;++k)if(lane+k*32<width){pa[k%8]+=qr[k]*ka[k];pc[k%8]+=qr[k]*kc[k];}
            float sa=((pa[0]+pa[4])+(pa[2]+pa[6]))+((pa[1]+pa[5])+(pa[3]+pa[7]));
            float sc=((pc[0]+pc[4])+(pc[2]+pc[6]))+((pc[1]+pc[5])+(pc[3]+pc[7]));
            for(unsigned int offset=16;offset>0;offset>>=1){
                sa+=__shfl_down_sync(0xffffffff,sa,offset);sc+=__shfl_down_sync(0xffffffff,sc,offset);}
            if(lane==0){scores[t-start]=sa*scale;if(second)scores[t+8-start]=sc*scale;}
        }
    }else
    for(unsigned int t=start+warp;t<end;t+=8){
        float part[8]={0.0f};
        #pragma unroll
        for(unsigned int group=0;group<8;++group){
            for(unsigned int c=lane+group*32;c<width;c+=256)
                part[group]+=q[(unsigned long long)head*width+c]*keys[t*stride+(unsigned long long)kv*width+c];
        }
        // Preserve the original 256-thread tree (strides 128,64,32).
        float sum=((part[0]+part[4])+(part[2]+part[6]))+((part[1]+part[5])+(part[3]+part[7]));
        for(unsigned int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(lane==0)scores[t-start]=sum*scale;
    }
    __syncthreads();
    if(ParallelExp && end-start>=64){
        __shared__ float maximum,total;
        if(thread==0){
            float m=-3.402823466e38F;
            for(unsigned int i=0;i<end-start;++i)m=fmaxf(m,scores[i]);
            maximum=m;
        }
        __syncthreads();
        for(unsigned int i=thread;i<end-start;i+=256)scores[i]=expf(scores[i]-maximum);
        __syncthreads();
        if(thread==0){
            float sum=0.0f;
            for(unsigned int i=0;i<end-start;++i)sum+=scores[i];
            total=sum;
        }
        __syncthreads();
        for(unsigned int i=thread;i<end-start;i+=256)scores[i]/=total;
    }else{
        // Same values as the original single-thread softmax: fmaxf is
        // order-independent, each expf/division is elementwise, and the
        // total keeps its sequential order (thread 0).
        __shared__ float maximum,total;
        const unsigned int n=end-start;
        if(warp==0){
            float m=-3.402823466e38F;
            for(unsigned int i=lane;i<n;i+=32)m=fmaxf(m,scores[i]);
            for(unsigned int offset=16;offset>0;offset>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,offset));
            if(lane==0)maximum=m;
        }
        __syncthreads();
        for(unsigned int i=thread;i<n;i+=256)scores[i]=expf(scores[i]-maximum);
        __syncthreads();
        if(thread==0){
            float sum=0.0f;
            for(unsigned int i=0;i<n;++i)sum+=scores[i];
            total=sum;
        }
        __syncthreads();
        for(unsigned int i=thread;i<n;i+=256)scores[i]/=total;
    }

    __syncthreads();
    for (unsigned int c=thread;c<width;c+=256) {
        float sum=0.0f;
        const float* vp=values+(unsigned long long)kv*width+c;
        unsigned int t=start;
        // Loads of eight tokens in flight; accumulation order unchanged.
        for(;t+8<=end;t+=8){
            float vv[8];
            #pragma unroll
            for(unsigned int u=0;u<8;++u)vv[u]=vp[(t+u)*stride];
            #pragma unroll
            for(unsigned int u=0;u<8;++u)sum+=scores[t+u-start]*vv[u];
        }
        for(;t<end;++t) sum+=scores[t-start]*vp[t*stride];
        output[(unsigned long long)head*width+c]=sum;
    }
}

extern "C" __global__ void causal_attention(const float* q,const float* keys,const float* values,float* output,unsigned int width,unsigned int qheads,unsigned int kvheads,unsigned int tokens,unsigned int position,unsigned int window,float scale){attention_impl<false>(q,keys,values,output,width,qheads,kvheads,tokens,position,window,scale);}
extern "C" __global__ void attention_dynamic(const float* q,const float* keys,const float* values,float* output,unsigned int width,unsigned int qheads,unsigned int kvheads,unsigned int window,float scale,const unsigned int* state){attention_impl<false>(q,keys,values,output,width,qheads,kvheads,state[1]+1,state[1],window,scale);}

extern "C" __global__ void causal_attention_parallel_exp(const float* q,const float* keys,const float* values,float* output,unsigned int width,unsigned int qheads,unsigned int kvheads,unsigned int tokens,unsigned int position,unsigned int window,float scale){attention_impl<true>(q,keys,values,output,width,qheads,kvheads,tokens,position,window,scale);}
extern "C" __global__ void attention_dynamic_parallel_exp(const float* q,const float* keys,const float* values,float* output,unsigned int width,unsigned int qheads,unsigned int kvheads,unsigned int window,float scale,const unsigned int* state){attention_impl<true>(q,keys,values,output,width,qheads,kvheads,state[1]+1,state[1],window,scale);}

// ---- Split attention for long contexts (bit-identical to attention_impl<false>) ----
// 1) scores: every token score is independent, so tokens are spread over many
//    blocks; the per-token arithmetic (q in registers, 8-group tree) is the
//    one used by attention_impl for widths <= 512.
// 2) softmax: max (order-independent fmaxf), elementwise expf, sequential sum
//    in index order by one thread, elementwise division.
// 3) values: one thread per (head, channel), accumulating tokens in order.
// scores is [q_heads][stride_scores]. tokens/position come from state when
// state != nullptr (graph path).
__device__ __forceinline__ void split_range(unsigned int tokens,unsigned int position,unsigned int window,
    unsigned int& start,unsigned int& end){
    end=min(tokens,position+1);
    start=(window>0 && end>window) ? end-window : 0;
}
extern "C" __global__ void attention_split_scores(const float* q,const float* keys,float* scores,
    unsigned int width,unsigned int q_heads,unsigned int kv_heads,unsigned int tokens,unsigned int position,
    unsigned int window,float scale,unsigned int stride_scores,const unsigned int* state){
    if(state){tokens=state[1]+1;position=state[1];}
    const unsigned int head=blockIdx.x,kv=head/(q_heads/kv_heads);
    unsigned int start,end;split_range(tokens,position,window,start,end);
    const unsigned int n=end-start,warp=threadIdx.x/32,lane=threadIdx.x&31;
    const unsigned int first=blockIdx.y*8+warp,step=gridDim.y*8;
    if(first>=n)return;
    const unsigned long long stride=(unsigned long long)kv_heads*width;
    float qr[16];
    #pragma unroll
    for(unsigned int k=0;k<16;++k){const unsigned int c=lane+k*32;qr[k]=c<width?q[(unsigned long long)head*width+c]:0.0f;}
    const float* kb=keys+(unsigned long long)kv*width;
    float* out=scores+(unsigned long long)head*stride_scores;
    for(unsigned int i=first;i<n;i+=2*step){
        const bool second=i+step<n;
        const unsigned long long t=start+i,t2=t+step;
        float ka[16],kc[16];
        #pragma unroll
        for(unsigned int k=0;k<16;++k){const unsigned int c=lane+k*32;
            ka[k]=c<width?kb[t*stride+c]:0.0f;kc[k]=(second&&c<width)?kb[t2*stride+c]:0.0f;}
        float pa[8]={0.0f},pc[8]={0.0f};
        #pragma unroll
        for(unsigned int k=0;k<16;++k)if(lane+k*32<width){pa[k%8]+=qr[k]*ka[k];pc[k%8]+=qr[k]*kc[k];}
        float sa=((pa[0]+pa[4])+(pa[2]+pa[6]))+((pa[1]+pa[5])+(pa[3]+pa[7]));
        float sc=((pc[0]+pc[4])+(pc[2]+pc[6]))+((pc[1]+pc[5])+(pc[3]+pc[7]));
        for(unsigned int offset=16;offset>0;offset>>=1){
            sa+=__shfl_down_sync(0xffffffff,sa,offset);sc+=__shfl_down_sync(0xffffffff,sc,offset);}
        if(lane==0){out[i]=sa*scale;if(second)out[i+step]=sc*scale;}
    }
}
extern "C" __global__ void attention_split_softmax(float* scores,unsigned int tokens,unsigned int position,
    unsigned int window,unsigned int stride_scores,const unsigned int* state){
    if(state){tokens=state[1]+1;position=state[1];}
    unsigned int start,end;split_range(tokens,position,window,start,end);
    const unsigned int n=end-start,thread=threadIdx.x,warp=thread/32,lane=thread&31;
    float* s=scores+(unsigned long long)blockIdx.x*stride_scores;
    __shared__ float part[8];
    __shared__ float maximum,total;
    float m=-3.402823466e38F;
    for(unsigned int i=thread;i<n;i+=256)m=fmaxf(m,s[i]);
    for(unsigned int offset=16;offset>0;offset>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,offset));
    if(lane==0)part[warp]=m;
    __syncthreads();
    if(thread==0){float r=part[0];for(int w=1;w<8;++w)r=fmaxf(r,part[w]);maximum=r;}
    __syncthreads();
    for(unsigned int i=thread;i<n;i+=256)s[i]=expf(s[i]-maximum);
    __syncthreads();
    if(thread==0){
        // Sequential in index order; 16-byte loads keep only the add chain.
        float sum=0.0f;unsigned int i=0;
        for(;i+4<=n;i+=4){const float4 v=*(const float4*)(s+i);sum+=v.x;sum+=v.y;sum+=v.z;sum+=v.w;}
        for(;i<n;++i)sum+=s[i];
        total=sum;
    }
    __syncthreads();
    for(unsigned int i=thread;i<n;i+=256)s[i]/=total;
}
extern "C" __global__ void attention_split_values(const float* scores,const float* values,float* output,
    unsigned int width,unsigned int q_heads,unsigned int kv_heads,unsigned int tokens,unsigned int position,
    unsigned int window,unsigned int stride_scores,const unsigned int* state){
    if(state){tokens=state[1]+1;position=state[1];}
    unsigned int start,end;split_range(tokens,position,window,start,end);
    const unsigned int head=blockIdx.x,c=blockIdx.y*64+threadIdx.x;
    if(c>=width)return;
    const unsigned int kv=head/(q_heads/kv_heads),n=end-start;
    const unsigned long long stride=(unsigned long long)kv_heads*width;
    const float* p=scores+(unsigned long long)head*stride_scores;
    const float* vp=values+(unsigned long long)kv*width+c+(unsigned long long)start*stride;
    float sum=0.0f;unsigned int i=0;
    // Sixteen tokens in flight; accumulation order unchanged.
    for(;i+16<=n;i+=16){
        float vv[16];
        #pragma unroll
        for(unsigned int u=0;u<16;++u)vv[u]=vp[(unsigned long long)(i+u)*stride];
        #pragma unroll
        for(unsigned int u=0;u<16;++u)sum+=p[i+u]*vv[u];
    }
    for(;i<n;++i)sum+=p[i]*vp[(unsigned long long)i*stride];
    output[(unsigned long long)head*width+c]=sum;
}
