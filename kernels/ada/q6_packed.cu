// Q6_K repacked at load time (lossless): per (row, block) 208 bytes =
// three u32 code words per lane [word][lane j], then the 16 int8 scales;
// d halves live in a separate array. Same values, same per-lane order and
// reduction tree as q6_project, so results are bit-identical.
template<bool WriteCap> __device__ __forceinline__ void q6_project_packed(const unsigned char* codes,const unsigned short* dv,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap) {
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
    #pragma unroll 2
    for(unsigned int block=lane/16;block<blocks;block+=2){
        const unsigned char* b=codes+((unsigned long long)row*blocks+block)*208;
        const unsigned int* cw=(const unsigned int*)b;
        const unsigned int c0=cw[j],c1=cw[16+j],c2=cw[32+j];
        const uint4 sv=*(const uint4*)(b+192);
        const unsigned int sc0=sv.x,sc1=sv.y,sc2=sv.z,sc3=sv.w;
        const float d=embedding_half(dv[(unsigned long long)row*blocks+block]);
        const unsigned int hi00=((c2>>0)&255u);
        const unsigned int lo000=((c0>>0)&255u);
        const unsigned int lo010=((c0>>16)&255u);
        const unsigned int hi01=((c2>>8)&255u);
        const unsigned int lo001=((c0>>8)&255u);
        const unsigned int lo011=((c0>>24)&255u);
        const int q0=(int)((lo000&15)|(((hi00>>0)&3)<<4))-32;
        s0+=((d*(float)(signed char)((sc0>>0)&255u))*(float)q0)*x[block*256+0+j];
        const int q1=(int)((lo001&15)|(((hi01>>0)&3)<<4))-32;
        s1+=((d*(float)(signed char)((sc0>>8)&255u))*(float)q1)*x[block*256+16+j];
        const int q2=(int)((lo010&15)|(((hi00>>2)&3)<<4))-32;
        s2+=((d*(float)(signed char)((sc0>>16)&255u))*(float)q2)*x[block*256+32+j];
        const int q3=(int)((lo011&15)|(((hi01>>2)&3)<<4))-32;
        s3+=((d*(float)(signed char)((sc0>>24)&255u))*(float)q3)*x[block*256+48+j];
        const int q4=(int)((lo000>>4)|(((hi00>>4)&3)<<4))-32;
        s4+=((d*(float)(signed char)((sc1>>0)&255u))*(float)q4)*x[block*256+64+j];
        const int q5=(int)((lo001>>4)|(((hi01>>4)&3)<<4))-32;
        s5+=((d*(float)(signed char)((sc1>>8)&255u))*(float)q5)*x[block*256+80+j];
        const int q6=(int)((lo010>>4)|(((hi00>>6)&3)<<4))-32;
        s6+=((d*(float)(signed char)((sc1>>16)&255u))*(float)q6)*x[block*256+96+j];
        const int q7=(int)((lo011>>4)|(((hi01>>6)&3)<<4))-32;
        s7+=((d*(float)(signed char)((sc1>>24)&255u))*(float)q7)*x[block*256+112+j];
        const unsigned int hi10=((c2>>16)&255u);
        const unsigned int lo100=((c1>>0)&255u);
        const unsigned int lo110=((c1>>16)&255u);
        const unsigned int hi11=((c2>>24)&255u);
        const unsigned int lo101=((c1>>8)&255u);
        const unsigned int lo111=((c1>>24)&255u);
        const int q8=(int)((lo100&15)|(((hi10>>0)&3)<<4))-32;
        s8+=((d*(float)(signed char)((sc2>>0)&255u))*(float)q8)*x[block*256+128+j];
        const int q9=(int)((lo101&15)|(((hi11>>0)&3)<<4))-32;
        s9+=((d*(float)(signed char)((sc2>>8)&255u))*(float)q9)*x[block*256+144+j];
        const int q10=(int)((lo110&15)|(((hi10>>2)&3)<<4))-32;
        s10+=((d*(float)(signed char)((sc2>>16)&255u))*(float)q10)*x[block*256+160+j];
        const int q11=(int)((lo111&15)|(((hi11>>2)&3)<<4))-32;
        s11+=((d*(float)(signed char)((sc2>>24)&255u))*(float)q11)*x[block*256+176+j];
        const int q12=(int)((lo100>>4)|(((hi10>>4)&3)<<4))-32;
        s12+=((d*(float)(signed char)((sc3>>0)&255u))*(float)q12)*x[block*256+192+j];
        const int q13=(int)((lo101>>4)|(((hi11>>4)&3)<<4))-32;
        s13+=((d*(float)(signed char)((sc3>>8)&255u))*(float)q13)*x[block*256+208+j];
        const int q14=(int)((lo110>>4)|(((hi10>>6)&3)<<4))-32;
        s14+=((d*(float)(signed char)((sc3>>16)&255u))*(float)q14)*x[block*256+224+j];
        const int q15=(int)((lo111>>4)|(((hi11>>6)&3)<<4))-32;
        s15+=((d*(float)(signed char)((sc3>>24)&255u))*(float)q15)*x[block*256+240+j];
    }
    float sum=((((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7)))+(((s8+s9)+(s10+s11))+((s12+s13)+(s14+s15))));
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0){raw[row]=sum;if(WriteCap)capped[row]=tanhf(sum/cap)*cap;}
}
extern "C" __global__ void q6_matvec_packed(const unsigned char* c,const unsigned short* d,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap){q6_project_packed<true>(c,d,x,raw,capped,columns,rows,cap);}
extern "C" __global__ void q6_matvec_raw_packed(const unsigned char* c,const unsigned short* d,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap){q6_project_packed<false>(c,d,x,raw,capped,columns,rows,cap);}
// Byte o (< 208) of the original Q6_K block, read from the packed block.
__device__ __forceinline__ unsigned int q6_packed_byte(const unsigned char* b,unsigned int o){
    if(o>=192)return b[o];
    const unsigned int k=o/16,lane=o%16;
    return b[(k/4)*64+lane*4+(k%4)];
}
__device__ __forceinline__ void lookup_packed_impl(const unsigned char* codes,const unsigned short* dv,float* output,unsigned int columns,unsigned int token){
    unsigned int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=columns)return;
    const unsigned long long index=(unsigned long long)token*(columns/256)+c/256;
    const unsigned char* b=codes+index*208;
    unsigned int j=c%256,g=j/128,q=(j%128)/32,l=j%32;
    unsigned int lo=q6_packed_byte(b,g*64+(q%2)*32+l);lo=q<2?(lo&15):(lo>>4);
    unsigned int hi=(q6_packed_byte(b,128+g*32+l)>>(q*2))&3;
    int code=(int)(lo|(hi<<4))-32;signed char s=(signed char)b[192+g*8+q*2+l/16];
    float d=embedding_half(dv[index]);
    output[c]=((d*(float)s)*(float)code)*sqrtf((float)columns);
}
extern "C" __global__ void lookup_q6_packed(const unsigned char* c,const unsigned short* d,float* output,unsigned int columns,unsigned int token){lookup_packed_impl(c,d,output,columns,token);}
extern "C" __global__ void lookup_dynamic_packed(const unsigned char* c,const unsigned short* d,float* output,unsigned int columns,const unsigned int* state){lookup_packed_impl(c,d,output,columns,state[0]);}

// Batched head (B tokens, weights read once). Same per-token arithmetic and
// tree as q6_project_packed: raw logits are bit-identical. x is [B][columns].
template<int B> __device__ __forceinline__ void q6_project_packed_batch(const unsigned char* codes,const unsigned short* dv,const float* x,float* raw,unsigned int columns,unsigned int rows) {
    const unsigned int lane=threadIdx.x&31,row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(row>=rows)return;
    const unsigned int j=lane&15,blocks=columns/256;
    float s0[B];
    #pragma unroll
    for(int t=0;t<B;++t)s0[t]=0.0f;
    float s1[B];
    #pragma unroll
    for(int t=0;t<B;++t)s1[t]=0.0f;
    float s2[B];
    #pragma unroll
    for(int t=0;t<B;++t)s2[t]=0.0f;
    float s3[B];
    #pragma unroll
    for(int t=0;t<B;++t)s3[t]=0.0f;
    float s4[B];
    #pragma unroll
    for(int t=0;t<B;++t)s4[t]=0.0f;
    float s5[B];
    #pragma unroll
    for(int t=0;t<B;++t)s5[t]=0.0f;
    float s6[B];
    #pragma unroll
    for(int t=0;t<B;++t)s6[t]=0.0f;
    float s7[B];
    #pragma unroll
    for(int t=0;t<B;++t)s7[t]=0.0f;
    float s8[B];
    #pragma unroll
    for(int t=0;t<B;++t)s8[t]=0.0f;
    float s9[B];
    #pragma unroll
    for(int t=0;t<B;++t)s9[t]=0.0f;
    float s10[B];
    #pragma unroll
    for(int t=0;t<B;++t)s10[t]=0.0f;
    float s11[B];
    #pragma unroll
    for(int t=0;t<B;++t)s11[t]=0.0f;
    float s12[B];
    #pragma unroll
    for(int t=0;t<B;++t)s12[t]=0.0f;
    float s13[B];
    #pragma unroll
    for(int t=0;t<B;++t)s13[t]=0.0f;
    float s14[B];
    #pragma unroll
    for(int t=0;t<B;++t)s14[t]=0.0f;
    float s15[B];
    #pragma unroll
    for(int t=0;t<B;++t)s15[t]=0.0f;
        for(unsigned int block=lane/16;block<blocks;block+=2){
        const unsigned char* b=codes+((unsigned long long)row*blocks+block)*208;
        const unsigned int* cw=(const unsigned int*)b;
        const unsigned int c0=cw[j],c1=cw[16+j],c2=cw[32+j];
        const uint4 sv=*(const uint4*)(b+192);
        const unsigned int sc0=sv.x,sc1=sv.y,sc2=sv.z,sc3=sv.w;
        const float d=embedding_half(dv[(unsigned long long)row*blocks+block]);
        const unsigned int hi00=((c2>>0)&255u);
        const unsigned int lo000=((c0>>0)&255u);
        const unsigned int lo010=((c0>>16)&255u);
        const unsigned int hi01=((c2>>8)&255u);
        const unsigned int lo001=((c0>>8)&255u);
        const unsigned int lo011=((c0>>24)&255u);
        const int q0=(int)((lo000&15)|(((hi00>>0)&3)<<4))-32;
        {const float w0=((d*(float)(signed char)((sc0>>0)&255u))*(float)q0);
        #pragma unroll
        for(int t=0;t<B;++t)s0[t]+=w0*x[(unsigned long long)t*columns+block*256+0+j];}
        const int q1=(int)((lo001&15)|(((hi01>>0)&3)<<4))-32;
        {const float w1=((d*(float)(signed char)((sc0>>8)&255u))*(float)q1);
        #pragma unroll
        for(int t=0;t<B;++t)s1[t]+=w1*x[(unsigned long long)t*columns+block*256+16+j];}
        const int q2=(int)((lo010&15)|(((hi00>>2)&3)<<4))-32;
        {const float w2=((d*(float)(signed char)((sc0>>16)&255u))*(float)q2);
        #pragma unroll
        for(int t=0;t<B;++t)s2[t]+=w2*x[(unsigned long long)t*columns+block*256+32+j];}
        const int q3=(int)((lo011&15)|(((hi01>>2)&3)<<4))-32;
        {const float w3=((d*(float)(signed char)((sc0>>24)&255u))*(float)q3);
        #pragma unroll
        for(int t=0;t<B;++t)s3[t]+=w3*x[(unsigned long long)t*columns+block*256+48+j];}
        const int q4=(int)((lo000>>4)|(((hi00>>4)&3)<<4))-32;
        {const float w4=((d*(float)(signed char)((sc1>>0)&255u))*(float)q4);
        #pragma unroll
        for(int t=0;t<B;++t)s4[t]+=w4*x[(unsigned long long)t*columns+block*256+64+j];}
        const int q5=(int)((lo001>>4)|(((hi01>>4)&3)<<4))-32;
        {const float w5=((d*(float)(signed char)((sc1>>8)&255u))*(float)q5);
        #pragma unroll
        for(int t=0;t<B;++t)s5[t]+=w5*x[(unsigned long long)t*columns+block*256+80+j];}
        const int q6=(int)((lo010>>4)|(((hi00>>6)&3)<<4))-32;
        {const float w6=((d*(float)(signed char)((sc1>>16)&255u))*(float)q6);
        #pragma unroll
        for(int t=0;t<B;++t)s6[t]+=w6*x[(unsigned long long)t*columns+block*256+96+j];}
        const int q7=(int)((lo011>>4)|(((hi01>>6)&3)<<4))-32;
        {const float w7=((d*(float)(signed char)((sc1>>24)&255u))*(float)q7);
        #pragma unroll
        for(int t=0;t<B;++t)s7[t]+=w7*x[(unsigned long long)t*columns+block*256+112+j];}
        const unsigned int hi10=((c2>>16)&255u);
        const unsigned int lo100=((c1>>0)&255u);
        const unsigned int lo110=((c1>>16)&255u);
        const unsigned int hi11=((c2>>24)&255u);
        const unsigned int lo101=((c1>>8)&255u);
        const unsigned int lo111=((c1>>24)&255u);
        const int q8=(int)((lo100&15)|(((hi10>>0)&3)<<4))-32;
        {const float w8=((d*(float)(signed char)((sc2>>0)&255u))*(float)q8);
        #pragma unroll
        for(int t=0;t<B;++t)s8[t]+=w8*x[(unsigned long long)t*columns+block*256+128+j];}
        const int q9=(int)((lo101&15)|(((hi11>>0)&3)<<4))-32;
        {const float w9=((d*(float)(signed char)((sc2>>8)&255u))*(float)q9);
        #pragma unroll
        for(int t=0;t<B;++t)s9[t]+=w9*x[(unsigned long long)t*columns+block*256+144+j];}
        const int q10=(int)((lo110&15)|(((hi10>>2)&3)<<4))-32;
        {const float w10=((d*(float)(signed char)((sc2>>16)&255u))*(float)q10);
        #pragma unroll
        for(int t=0;t<B;++t)s10[t]+=w10*x[(unsigned long long)t*columns+block*256+160+j];}
        const int q11=(int)((lo111&15)|(((hi11>>2)&3)<<4))-32;
        {const float w11=((d*(float)(signed char)((sc2>>24)&255u))*(float)q11);
        #pragma unroll
        for(int t=0;t<B;++t)s11[t]+=w11*x[(unsigned long long)t*columns+block*256+176+j];}
        const int q12=(int)((lo100>>4)|(((hi10>>4)&3)<<4))-32;
        {const float w12=((d*(float)(signed char)((sc3>>0)&255u))*(float)q12);
        #pragma unroll
        for(int t=0;t<B;++t)s12[t]+=w12*x[(unsigned long long)t*columns+block*256+192+j];}
        const int q13=(int)((lo101>>4)|(((hi11>>4)&3)<<4))-32;
        {const float w13=((d*(float)(signed char)((sc3>>8)&255u))*(float)q13);
        #pragma unroll
        for(int t=0;t<B;++t)s13[t]+=w13*x[(unsigned long long)t*columns+block*256+208+j];}
        const int q14=(int)((lo110>>4)|(((hi10>>6)&3)<<4))-32;
        {const float w14=((d*(float)(signed char)((sc3>>16)&255u))*(float)q14);
        #pragma unroll
        for(int t=0;t<B;++t)s14[t]+=w14*x[(unsigned long long)t*columns+block*256+224+j];}
        const int q15=(int)((lo111>>4)|(((hi11>>6)&3)<<4))-32;
        {const float w15=((d*(float)(signed char)((sc3>>24)&255u))*(float)q15);
        #pragma unroll
        for(int t=0;t<B;++t)s15[t]+=w15*x[(unsigned long long)t*columns+block*256+240+j];}
    }
    #pragma unroll
    for(int t=0;t<B;++t){
        float sum=((((s0[t]+s1[t])+(s2[t]+s3[t]))+((s4[t]+s5[t])+(s6[t]+s7[t])))+(((s8[t]+s9[t])+(s10[t]+s11[t]))+((s12[t]+s13[t])+(s14[t]+s15[t]))));
        for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(lane==0)raw[(unsigned long long)t*rows+row]=sum;
    }
}
extern "C" __global__ void q6_matvec_raw_packed_b1(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch<1>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed_b2(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch<2>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed_b3(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch<3>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed_b4(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch<4>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed_b5(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch<5>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed_b6(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch<6>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed_b7(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch<7>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed_b8(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch<8>(c,d,x,raw,columns,rows);}
