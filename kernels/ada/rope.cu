// Rotation NeoX entre les deux moitiés d’une tête, une paire par thread.
// Les facteurs proportionnels GGUF sont déjà incorporés aux fréquences FP64.
__device__ __forceinline__ void rope_impl(const float* input, const double* inv_freq,
    float* output, unsigned int width, unsigned int elements, unsigned int position)
{
    const unsigned int pair=blockIdx.x*blockDim.x+threadIdx.x;
    if (pair>=elements/2) return;
    const unsigned int half=width/2;
    const unsigned int head=pair/half;
    const unsigned int i=pair%half;
    const unsigned int a=head*width+i;
    const unsigned int b=a+half;
    double sine,cosine;
    sincos((double)position*inv_freq[i],&sine,&cosine);
    const float s=(float)sine,c=(float)cosine;
    const float x=input[a],y=input[b];
    output[a]=x*c-y*s;
    output[b]=x*s+y*c;
}

extern "C" __global__ void rope_neox(const float* input,const double* freq,float* output,unsigned int width,unsigned int elements,unsigned int position){rope_impl(input,freq,output,width,elements,position);}
extern "C" __global__ void rope_dynamic(const float* input,const double* freq,float* output,unsigned int width,unsigned int elements,const unsigned int* state){rope_impl(input,freq,output,width,elements,state[1]);}
extern "C" __global__ void cache_dynamic(const float* key,const float* value,float* keys,float* values,unsigned int elements,const unsigned int* state){
    unsigned int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=elements)return;
    unsigned long long offset=(unsigned long long)state[1]*elements+c;keys[offset]=key[c];values[offset]=value[c];
}

// One block per head; identical RMS reduction and an explicit FP32 boundary
// in shared memory before NeoX rotation. Only widths 256 and 512 are launched.
__device__ void norm_rope_impl(const float* input,const float* weights,const double* freq,float* output,unsigned int width,float eps,unsigned int position,unsigned int head,bool rotate){
    __shared__ float partial[32];
    __shared__ float normalized[512];
    const unsigned int thread=threadIdx.x;
    const unsigned int lane=(thread&7)*32+(thread>>3);
    const unsigned long long base=(unsigned long long)head*width;
    float sum=0.0f;
    for(unsigned int i=lane;i<width;i+=256){float v=input[base+i];sum+=v*v;}
    for(unsigned int offset=4;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset,8);
    if((thread&7)==0)partial[thread>>3]=sum;
    __syncthreads();
    if(thread<32){
        sum=partial[thread];
        for(unsigned int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(thread==0)partial[0]=sum;
    }
    __syncthreads();
    const float inv=rsqrtf(partial[0]/(float)width+eps);
    for(unsigned int i=lane;i<width;i+=256)normalized[i]=(input[base+i]*inv)*weights[i];
    if(!rotate){
        for(unsigned int i=lane;i<width;i+=256)output[base+i]=normalized[i];
        return;
    }
    __syncthreads();
    const unsigned int half=width/2;
    if(thread<half){
        double sine,cosine;sincos((double)position*freq[thread],&sine,&cosine);
        const float s=(float)sine,c=(float)cosine;
        const float x=normalized[thread],y=normalized[thread+half];
        output[base+thread]=x*c-y*s;
        output[base+thread+half]=x*s+y*c;
    }
}
extern "C" __global__ void norm_rope(const float* input,const float* weights,const double* freq,float* output,unsigned int width,float eps,unsigned int position){norm_rope_impl(input,weights,freq,output,width,eps,position,blockIdx.x,true);}
extern "C" __global__ void norm_rope_dynamic(const float* input,const float* weights,const double* freq,float* output,unsigned int width,float eps,const unsigned int* state){norm_rope_impl(input,weights,freq,output,width,eps,state[1],blockIdx.x,true);}

__device__ void prepare_qkv_impl(const float* q,const float* k,const float* v,const float* qw,const float* kw,const float* vw,const double* freq,float* qo,float* ko,float* vo,unsigned int width,unsigned int qheads,unsigned int kvheads,float qe,float ke,float ve,unsigned int position){
    unsigned int h=blockIdx.x;
    if(h<qheads)norm_rope_impl(q,qw,freq,qo,width,qe,position,h,true);
    else if(h<qheads+kvheads)norm_rope_impl(k,kw,freq,ko,width,ke,position,h-qheads,true);
    else norm_rope_impl(v,vw,freq,vo,width,ve,position,h-qheads-kvheads,false);
}
extern "C" __global__ void prepare_qkv(const float* q,const float* k,const float* v,const float* qw,const float* kw,const float* vw,const double* freq,float* qo,float* ko,float* vo,unsigned int width,unsigned int qheads,unsigned int kvheads,float qe,float ke,float ve,unsigned int position){prepare_qkv_impl(q,k,v,qw,kw,vw,freq,qo,ko,vo,width,qheads,kvheads,qe,ke,ve,position);}
extern "C" __global__ void prepare_qkv_dynamic(const float* q,const float* k,const float* v,const float* qw,const float* kw,const float* vw,const double* freq,float* qo,float* ko,float* vo,unsigned int width,unsigned int qheads,unsigned int kvheads,float qe,float ke,float ve,const unsigned int* state){prepare_qkv_impl(q,k,v,qw,kw,vw,freq,qo,ko,vo,width,qheads,kvheads,qe,ke,ve,state[1]);}

// Graph-only: write prepared K/V straight into the current cache slot.
extern "C" __global__ void prepare_qkv_cache_dynamic(const float* q,const float* k,const float* v,const float* qw,const float* kw,const float* vw,const double* freq,float* qo,float* keys,float* values,unsigned int width,unsigned int qheads,unsigned int kvheads,float qe,float ke,float ve,const unsigned int* state){
    unsigned long long offset=(unsigned long long)state[1]*kvheads*width;
    prepare_qkv_impl(q,k,v,qw,kw,vw,freq,qo,keys+offset,values+offset,width,qheads,kvheads,qe,ke,ve,state[1]);
}
