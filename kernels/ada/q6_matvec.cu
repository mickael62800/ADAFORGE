template<bool WriteCap> __device__ __forceinline__ void q6_project(const unsigned char* weights,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap) {
    const unsigned int lane=threadIdx.x&31,row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(row>=rows)return;
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
        const unsigned char* b=weights+((unsigned long long)row*blocks+block)*210;
        const float d=embedding_half(*(const unsigned short*)(b+208));
        const unsigned int hi00=b[128+j];
        const unsigned int lo000=b[0+j];
        const unsigned int lo010=b[32+j];
        const unsigned int hi01=b[144+j];
        const unsigned int lo001=b[16+j];
        const unsigned int lo011=b[48+j];
        const int q0=(int)((lo000&15)|(((hi00>>0)&3)<<4))-32;
        s0+=((d*(float)(signed char)b[192])*(float)q0)*x[block*256+0+j];
        const int q1=(int)((lo001&15)|(((hi01>>0)&3)<<4))-32;
        s1+=((d*(float)(signed char)b[193])*(float)q1)*x[block*256+16+j];
        const int q2=(int)((lo010&15)|(((hi00>>2)&3)<<4))-32;
        s2+=((d*(float)(signed char)b[194])*(float)q2)*x[block*256+32+j];
        const int q3=(int)((lo011&15)|(((hi01>>2)&3)<<4))-32;
        s3+=((d*(float)(signed char)b[195])*(float)q3)*x[block*256+48+j];
        const int q4=(int)((lo000>>4)|(((hi00>>4)&3)<<4))-32;
        s4+=((d*(float)(signed char)b[196])*(float)q4)*x[block*256+64+j];
        const int q5=(int)((lo001>>4)|(((hi01>>4)&3)<<4))-32;
        s5+=((d*(float)(signed char)b[197])*(float)q5)*x[block*256+80+j];
        const int q6=(int)((lo010>>4)|(((hi00>>6)&3)<<4))-32;
        s6+=((d*(float)(signed char)b[198])*(float)q6)*x[block*256+96+j];
        const int q7=(int)((lo011>>4)|(((hi01>>6)&3)<<4))-32;
        s7+=((d*(float)(signed char)b[199])*(float)q7)*x[block*256+112+j];
        const unsigned int hi10=b[160+j];
        const unsigned int lo100=b[64+j];
        const unsigned int lo110=b[96+j];
        const unsigned int hi11=b[176+j];
        const unsigned int lo101=b[80+j];
        const unsigned int lo111=b[112+j];
        const int q8=(int)((lo100&15)|(((hi10>>0)&3)<<4))-32;
        s8+=((d*(float)(signed char)b[200])*(float)q8)*x[block*256+128+j];
        const int q9=(int)((lo101&15)|(((hi11>>0)&3)<<4))-32;
        s9+=((d*(float)(signed char)b[201])*(float)q9)*x[block*256+144+j];
        const int q10=(int)((lo110&15)|(((hi10>>2)&3)<<4))-32;
        s10+=((d*(float)(signed char)b[202])*(float)q10)*x[block*256+160+j];
        const int q11=(int)((lo111&15)|(((hi11>>2)&3)<<4))-32;
        s11+=((d*(float)(signed char)b[203])*(float)q11)*x[block*256+176+j];
        const int q12=(int)((lo100>>4)|(((hi10>>4)&3)<<4))-32;
        s12+=((d*(float)(signed char)b[204])*(float)q12)*x[block*256+192+j];
        const int q13=(int)((lo101>>4)|(((hi11>>4)&3)<<4))-32;
        s13+=((d*(float)(signed char)b[205])*(float)q13)*x[block*256+208+j];
        const int q14=(int)((lo110>>4)|(((hi10>>6)&3)<<4))-32;
        s14+=((d*(float)(signed char)b[206])*(float)q14)*x[block*256+224+j];
        const int q15=(int)((lo111>>4)|(((hi11>>6)&3)<<4))-32;
        s15+=((d*(float)(signed char)b[207])*(float)q15)*x[block*256+240+j];
    }
    float sum=((((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7)))+(((s8+s9)+(s10+s11))+((s12+s13)+(s14+s15))));
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0){raw[row]=sum;if(WriteCap)capped[row]=tanhf(sum/cap)*cap;}
}
extern "C" __global__ void q6_matvec(const unsigned char* w,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap){q6_project<true>(w,x,raw,capped,columns,rows,cap);}
extern "C" __global__ void q6_matvec_raw(const unsigned char* w,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap){q6_project<false>(w,x,raw,capped,columns,rows,cap);}
__device__ __forceinline__ void lookup_impl(const unsigned char* weights,float* output,unsigned int columns,unsigned int token) {
    unsigned int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=columns)return;
    const unsigned char* b=weights+((unsigned long long)token*(columns/256)+c/256)*210;
    unsigned int j=c%256,g=j/128,q=(j%128)/32,l=j%32;
    unsigned int lo=b[g*64+(q%2)*32+l];lo=q<2?(lo&15):(lo>>4);
    unsigned int hi=(b[128+g*32+l]>>(q*2))&3;
    int code=(int)(lo|(hi<<4))-32;signed char s=(signed char)b[192+g*8+q*2+l/16];
    float d=embedding_half((unsigned int)b[208]|((unsigned int)b[209]<<8));
    output[c]=((d*(float)s)*(float)code)*sqrtf((float)columns);
}

extern "C" __global__ void lookup_q6(const unsigned char* weights,float* output,unsigned int columns,unsigned int token){lookup_impl(weights,output,columns,token);}
extern "C" __global__ void lookup_dynamic(const unsigned char* weights,float* output,unsigned int columns,const unsigned int* state){lookup_impl(weights,output,columns,state[0]);}
