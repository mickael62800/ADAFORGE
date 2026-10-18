// Greedy exact sur logits FP32 : maximum, puis plus petit ID ; refus des non-finis.
__device__ void greedy_reduce(float* values,unsigned int* ids,unsigned int* bad) {
    unsigned int t=threadIdx.x;
    __syncthreads();
    for(unsigned int stride=128;stride>0;stride>>=1){
        if(t<stride){
            float v=values[t+stride];unsigned int i=ids[t+stride];
            if(v>values[t] || (v==values[t] && i<ids[t])){values[t]=v;ids[t]=i;}
            bad[t]|=bad[t+stride];
        }
        __syncthreads();
    }
}
extern "C" __global__ void greedy_partial(const float* logits,float* partial_values,unsigned int* partial_ids,unsigned int* partial_bad,unsigned int count){
    __shared__ float values[256];__shared__ unsigned int ids[256],bad[256];
    unsigned int t=threadIdx.x,i=blockIdx.x*256+t;
    float v=i<count?logits[i]:-__int_as_float(0x7f800000);
    bad[t]=(i<count && !isfinite(v))?1:0;
    values[t]=bad[t]?-__int_as_float(0x7f800000):v;ids[t]=i<count?i:0xffffffff;
    greedy_reduce(values,ids,bad);
    if(t==0){partial_values[blockIdx.x]=values[0];partial_ids[blockIdx.x]=ids[0];partial_bad[blockIdx.x]=bad[0];}
}
extern "C" __global__ void greedy_final(const float* partial_values,const unsigned int* partial_ids,const unsigned int* partial_bad,unsigned int* result,unsigned int count){
    __shared__ float values[256];__shared__ unsigned int ids[256],bad[256];
    unsigned int t=threadIdx.x;float best=-__int_as_float(0x7f800000);unsigned int id=0xffffffff,invalid=0;
    for(unsigned int i=t;i<count;i+=256){
        float v=partial_values[i];unsigned int next=partial_ids[i];
        if(v>best || (v==best && next<id)){best=v;id=next;}
        invalid|=partial_bad[i];
    }
    values[t]=best;ids[t]=id;bad[t]=invalid;greedy_reduce(values,ids,bad);
    if(t==0){result[0]=ids[0];result[1]=bad[0];}
}
