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
    if (thread==0) {
        float maximum=-3.402823466e38F;
        for (unsigned int i=0;i<end-start;++i) maximum=fmaxf(maximum,scores[i]);
        float total=0.0f;
        for (unsigned int i=0;i<end-start;++i) {scores[i]=expf(scores[i]-maximum);total+=scores[i];}
        for (unsigned int i=0;i<end-start;++i) scores[i]/=total;
    }    }

    __syncthreads();
    for (unsigned int c=thread;c<width;c+=256) {
        float sum=0.0f;
        for (unsigned int t=start;t<end;++t) sum+=scores[t-start]*values[t*stride+(unsigned long long)kv*width+c];
        output[(unsigned long long)head*width+c]=sum;
    }
}

extern "C" __global__ void causal_attention(const float* q,const float* keys,const float* values,float* output,unsigned int width,unsigned int qheads,unsigned int kvheads,unsigned int tokens,unsigned int position,unsigned int window,float scale){attention_impl<false>(q,keys,values,output,width,qheads,kvheads,tokens,position,window,scale);}
extern "C" __global__ void attention_dynamic(const float* q,const float* keys,const float* values,float* output,unsigned int width,unsigned int qheads,unsigned int kvheads,unsigned int window,float scale,const unsigned int* state){attention_impl<false>(q,keys,values,output,width,qheads,kvheads,state[1]+1,state[1],window,scale);}

extern "C" __global__ void causal_attention_parallel_exp(const float* q,const float* keys,const float* values,float* output,unsigned int width,unsigned int qheads,unsigned int kvheads,unsigned int tokens,unsigned int position,unsigned int window,float scale){attention_impl<true>(q,keys,values,output,width,qheads,kvheads,tokens,position,window,scale);}
extern "C" __global__ void attention_dynamic_parallel_exp(const float* q,const float* keys,const float* values,float* output,unsigned int width,unsigned int qheads,unsigned int kvheads,unsigned int window,float scale,const unsigned int* state){attention_impl<true>(q,keys,values,output,width,qheads,kvheads,state[1]+1,state[1],window,scale);}
