// Lossless 4-row interleave. Original 32-lane reduction tree is preserved.
template<bool WriteCap> __device__ __forceinline__ void q6_project_tiled(const unsigned char* weights,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap) {
    const unsigned int lane=threadIdx.x/4,component=threadIdx.x&3,row=blockIdx.x*4+component;
    const unsigned int j=lane&15,blocks=columns/256;
    float s0=0.0f;
    float s1=0.0f;
    float s2=0.0f;
    float s3=0.0f;
    float s4=0.0f;
    float s5=0.0f;
    float s6=0.0f;
    float s7=0.0f;
    float s8=0.0f;
    float s9=0.0f;
    float s10=0.0f;
    float s11=0.0f;
    float s12=0.0f;
    float s13=0.0f;
    float s14=0.0f;
    float s15=0.0f;
    for(unsigned int block=lane/16;block<blocks;block+=2){
        const unsigned char* b=weights+((unsigned long long)blockIdx.x*blocks+block)*840;
        const float d=embedding_half(*(const unsigned short*)(b+832+component*2));
        const unsigned int hi00=b[(128+j)*4+component];
        const unsigned int lo000=b[(0+j)*4+component];
        const unsigned int lo010=b[(32+j)*4+component];
        const unsigned int hi01=b[(144+j)*4+component];
        const unsigned int lo001=b[(16+j)*4+component];
        const unsigned int lo011=b[(48+j)*4+component];
        const int q0=(int)((lo000&15)|(((hi00>>0)&3)<<4))-32;
        s0+=((d*(float)(signed char)b[(192)*4+component])*(float)q0)*x[block*256+0+j];
        const int q1=(int)((lo001&15)|(((hi01>>0)&3)<<4))-32;
        s1+=((d*(float)(signed char)b[(193)*4+component])*(float)q1)*x[block*256+16+j];
        const int q2=(int)((lo010&15)|(((hi00>>2)&3)<<4))-32;
        s2+=((d*(float)(signed char)b[(194)*4+component])*(float)q2)*x[block*256+32+j];
        const int q3=(int)((lo011&15)|(((hi01>>2)&3)<<4))-32;
        s3+=((d*(float)(signed char)b[(195)*4+component])*(float)q3)*x[block*256+48+j];
        const int q4=(int)((lo000>>4)|(((hi00>>4)&3)<<4))-32;
        s4+=((d*(float)(signed char)b[(196)*4+component])*(float)q4)*x[block*256+64+j];
        const int q5=(int)((lo001>>4)|(((hi01>>4)&3)<<4))-32;
        s5+=((d*(float)(signed char)b[(197)*4+component])*(float)q5)*x[block*256+80+j];
        const int q6=(int)((lo010>>4)|(((hi00>>6)&3)<<4))-32;
        s6+=((d*(float)(signed char)b[(198)*4+component])*(float)q6)*x[block*256+96+j];
        const int q7=(int)((lo011>>4)|(((hi01>>6)&3)<<4))-32;
        s7+=((d*(float)(signed char)b[(199)*4+component])*(float)q7)*x[block*256+112+j];
        const unsigned int hi10=b[(160+j)*4+component];
        const unsigned int lo100=b[(64+j)*4+component];
        const unsigned int lo110=b[(96+j)*4+component];
        const unsigned int hi11=b[(176+j)*4+component];
        const unsigned int lo101=b[(80+j)*4+component];
        const unsigned int lo111=b[(112+j)*4+component];
        const int q8=(int)((lo100&15)|(((hi10>>0)&3)<<4))-32;
        s8+=((d*(float)(signed char)b[(200)*4+component])*(float)q8)*x[block*256+128+j];
        const int q9=(int)((lo101&15)|(((hi11>>0)&3)<<4))-32;
        s9+=((d*(float)(signed char)b[(201)*4+component])*(float)q9)*x[block*256+144+j];
        const int q10=(int)((lo110&15)|(((hi10>>2)&3)<<4))-32;
        s10+=((d*(float)(signed char)b[(202)*4+component])*(float)q10)*x[block*256+160+j];
        const int q11=(int)((lo111&15)|(((hi11>>2)&3)<<4))-32;
        s11+=((d*(float)(signed char)b[(203)*4+component])*(float)q11)*x[block*256+176+j];
        const int q12=(int)((lo100>>4)|(((hi10>>4)&3)<<4))-32;
        s12+=((d*(float)(signed char)b[(204)*4+component])*(float)q12)*x[block*256+192+j];
        const int q13=(int)((lo101>>4)|(((hi11>>4)&3)<<4))-32;
        s13+=((d*(float)(signed char)b[(205)*4+component])*(float)q13)*x[block*256+208+j];
        const int q14=(int)((lo110>>4)|(((hi10>>6)&3)<<4))-32;
        s14+=((d*(float)(signed char)b[(206)*4+component])*(float)q14)*x[block*256+224+j];
        const int q15=(int)((lo111>>4)|(((hi11>>6)&3)<<4))-32;
        s15+=((d*(float)(signed char)b[(207)*4+component])*(float)q15)*x[block*256+240+j];
    }
    float sum=((((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7)))+(((s8+s9)+(s10+s11))+((s12+s13)+(s14+s15))));
    __shared__ float partial[128];
    partial[threadIdx.x]=sum;
    __syncthreads();
    if(threadIdx.x<32){
        const unsigned int t=threadIdx.x;
        sum=(partial[t]+partial[t+64])+(partial[t+32]+partial[t+96]);
        for(int offset=16;offset>=4;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(t<4 && row<rows){raw[row]=sum;if(WriteCap)capped[row]=tanhf(sum/cap)*cap;}
    }
}

extern "C" __global__ void q6_matvec_tiled(const unsigned char* w,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap){q6_project_tiled<true>(w,x,raw,capped,columns,rows,cap);}
extern "C" __global__ void q6_matvec_raw_tiled(const unsigned char* w,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap){q6_project_tiled<false>(w,x,raw,capped,columns,rows,cap);}
__device__ void lookup_tiled_impl(const unsigned char* weights,float* output,unsigned int columns,unsigned int token){
    const unsigned int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=columns)return;
    const unsigned int row=token&3;
    const unsigned char* b=weights+((unsigned long long)(token/4)*(columns/256)+c/256)*840;
    unsigned int j=c%256,g=j/128,q=(j%128)/32,l=j%32;
    unsigned int lo=b[(g*64+(q%2)*32+l)*4+row];lo=q<2?(lo&15):(lo>>4);
    unsigned int hi=(b[(128+g*32+l)*4+row]>>(q*2))&3;
    int code=(int)(lo|(hi<<4))-32;
    signed char s=(signed char)b[(192+g*8+q*2+l/16)*4+row];
    float d=embedding_half(*(const unsigned short*)(b+832+row*2));
    output[c]=((d*(float)s)*(float)code)*sqrtf((float)columns);
}
extern "C" __global__ void lookup_q6_tiled(const unsigned char* weights,float* output,unsigned int columns,unsigned int token){lookup_tiled_impl(weights,output,columns,token);}
extern "C" __global__ void lookup_dynamic_tiled(const unsigned char* weights,float* output,unsigned int columns,const unsigned int* state){lookup_tiled_impl(weights,output,columns,state[0]);}
