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

// Two rows per warp (rowA, rowB = rowA + rows/2 halves of the grid), loads of
// both in flight together. Per row: same values, order and tree as
// q6_project_packed (raw logits only): bit-identical.
__device__ __forceinline__ void q6_project_packed_dual(const unsigned char* codes,const unsigned short* dv,const float* x,float* raw,unsigned int columns,unsigned int rows) {
    const unsigned int lane=threadIdx.x&31,warps=gridDim.x*(blockDim.x/32);
    const unsigned int rowA=blockIdx.x*(blockDim.x/32)+threadIdx.x/32,rowB=rowA+warps;
    if(rowA>=rows)return;
    const bool hasB=rowB<rows;
    const unsigned int rowBc=hasB?rowB:rowA;
    const unsigned int j=lane&15,blocks=columns/256;
    float sa0=0.0f,sb0=0.0f;
    float sa1=0.0f,sb1=0.0f;
    float sa2=0.0f,sb2=0.0f;
    float sa3=0.0f,sb3=0.0f;
    float sa4=0.0f,sb4=0.0f;
    float sa5=0.0f,sb5=0.0f;
    float sa6=0.0f,sb6=0.0f;
    float sa7=0.0f,sb7=0.0f;
    float sa8=0.0f,sb8=0.0f;
    float sa9=0.0f,sb9=0.0f;
    float sa10=0.0f,sb10=0.0f;
    float sa11=0.0f,sb11=0.0f;
    float sa12=0.0f,sb12=0.0f;
    float sa13=0.0f,sb13=0.0f;
    float sa14=0.0f,sb14=0.0f;
    float sa15=0.0f,sb15=0.0f;
    for(unsigned int block=lane/16;block<blocks;block+=2){

        const unsigned char* ba=codes+((unsigned long long)rowA*blocks+block)*208;
        const unsigned int* cwa=(const unsigned int*)ba;
        const unsigned int c0a=cwa[j],c1a=cwa[16+j],c2a=cwa[32+j];
        const uint4 sva=*(const uint4*)(ba+192);
        const unsigned int sc0a=sva.x,sc1a=sva.y,sc2a=sva.z,sc3a=sva.w;
        const float da=embedding_half(dv[(unsigned long long)rowA*blocks+block]);
        const unsigned int hi00a=((c2a>>0)&255u);
        const unsigned int lo000a=((c0a>>0)&255u);
        const unsigned int lo010a=((c0a>>16)&255u);
        const unsigned int hi01a=((c2a>>8)&255u);
        const unsigned int lo001a=((c0a>>8)&255u);
        const unsigned int lo011a=((c0a>>24)&255u);
        const int q0a=(int)((lo000a&15)|(((hi00a>>0)&3)<<4))-32;
        sa0+=((da*(float)(signed char)((sc0a>>0)&255u))*(float)q0a)*x[block*256+0+j];
        const int q1a=(int)((lo001a&15)|(((hi01a>>0)&3)<<4))-32;
        sa1+=((da*(float)(signed char)((sc0a>>8)&255u))*(float)q1a)*x[block*256+16+j];
        const int q2a=(int)((lo010a&15)|(((hi00a>>2)&3)<<4))-32;
        sa2+=((da*(float)(signed char)((sc0a>>16)&255u))*(float)q2a)*x[block*256+32+j];
        const int q3a=(int)((lo011a&15)|(((hi01a>>2)&3)<<4))-32;
        sa3+=((da*(float)(signed char)((sc0a>>24)&255u))*(float)q3a)*x[block*256+48+j];
        const int q4a=(int)((lo000a>>4)|(((hi00a>>4)&3)<<4))-32;
        sa4+=((da*(float)(signed char)((sc1a>>0)&255u))*(float)q4a)*x[block*256+64+j];
        const int q5a=(int)((lo001a>>4)|(((hi01a>>4)&3)<<4))-32;
        sa5+=((da*(float)(signed char)((sc1a>>8)&255u))*(float)q5a)*x[block*256+80+j];
        const int q6a=(int)((lo010a>>4)|(((hi00a>>6)&3)<<4))-32;
        sa6+=((da*(float)(signed char)((sc1a>>16)&255u))*(float)q6a)*x[block*256+96+j];
        const int q7a=(int)((lo011a>>4)|(((hi01a>>6)&3)<<4))-32;
        sa7+=((da*(float)(signed char)((sc1a>>24)&255u))*(float)q7a)*x[block*256+112+j];
        const unsigned int hi10a=((c2a>>16)&255u);
        const unsigned int lo100a=((c1a>>0)&255u);
        const unsigned int lo110a=((c1a>>16)&255u);
        const unsigned int hi11a=((c2a>>24)&255u);
        const unsigned int lo101a=((c1a>>8)&255u);
        const unsigned int lo111a=((c1a>>24)&255u);
        const int q8a=(int)((lo100a&15)|(((hi10a>>0)&3)<<4))-32;
        sa8+=((da*(float)(signed char)((sc2a>>0)&255u))*(float)q8a)*x[block*256+128+j];
        const int q9a=(int)((lo101a&15)|(((hi11a>>0)&3)<<4))-32;
        sa9+=((da*(float)(signed char)((sc2a>>8)&255u))*(float)q9a)*x[block*256+144+j];
        const int q10a=(int)((lo110a&15)|(((hi10a>>2)&3)<<4))-32;
        sa10+=((da*(float)(signed char)((sc2a>>16)&255u))*(float)q10a)*x[block*256+160+j];
        const int q11a=(int)((lo111a&15)|(((hi11a>>2)&3)<<4))-32;
        sa11+=((da*(float)(signed char)((sc2a>>24)&255u))*(float)q11a)*x[block*256+176+j];
        const int q12a=(int)((lo100a>>4)|(((hi10a>>4)&3)<<4))-32;
        sa12+=((da*(float)(signed char)((sc3a>>0)&255u))*(float)q12a)*x[block*256+192+j];
        const int q13a=(int)((lo101a>>4)|(((hi11a>>4)&3)<<4))-32;
        sa13+=((da*(float)(signed char)((sc3a>>8)&255u))*(float)q13a)*x[block*256+208+j];
        const int q14a=(int)((lo110a>>4)|(((hi10a>>6)&3)<<4))-32;
        sa14+=((da*(float)(signed char)((sc3a>>16)&255u))*(float)q14a)*x[block*256+224+j];
        const int q15a=(int)((lo111a>>4)|(((hi11a>>6)&3)<<4))-32;
        sa15+=((da*(float)(signed char)((sc3a>>24)&255u))*(float)q15a)*x[block*256+240+j];
    

        const unsigned char* bb=codes+((unsigned long long)rowBc*blocks+block)*208;
        const unsigned int* cwb=(const unsigned int*)bb;
        const unsigned int c0b=cwb[j],c1b=cwb[16+j],c2b=cwb[32+j];
        const uint4 svb=*(const uint4*)(bb+192);
        const unsigned int sc0b=svb.x,sc1b=svb.y,sc2b=svb.z,sc3b=svb.w;
        const float db=embedding_half(dv[(unsigned long long)rowBc*blocks+block]);
        const unsigned int hi00b=((c2b>>0)&255u);
        const unsigned int lo000b=((c0b>>0)&255u);
        const unsigned int lo010b=((c0b>>16)&255u);
        const unsigned int hi01b=((c2b>>8)&255u);
        const unsigned int lo001b=((c0b>>8)&255u);
        const unsigned int lo011b=((c0b>>24)&255u);
        const int q0b=(int)((lo000b&15)|(((hi00b>>0)&3)<<4))-32;
        sb0+=((db*(float)(signed char)((sc0b>>0)&255u))*(float)q0b)*x[block*256+0+j];
        const int q1b=(int)((lo001b&15)|(((hi01b>>0)&3)<<4))-32;
        sb1+=((db*(float)(signed char)((sc0b>>8)&255u))*(float)q1b)*x[block*256+16+j];
        const int q2b=(int)((lo010b&15)|(((hi00b>>2)&3)<<4))-32;
        sb2+=((db*(float)(signed char)((sc0b>>16)&255u))*(float)q2b)*x[block*256+32+j];
        const int q3b=(int)((lo011b&15)|(((hi01b>>2)&3)<<4))-32;
        sb3+=((db*(float)(signed char)((sc0b>>24)&255u))*(float)q3b)*x[block*256+48+j];
        const int q4b=(int)((lo000b>>4)|(((hi00b>>4)&3)<<4))-32;
        sb4+=((db*(float)(signed char)((sc1b>>0)&255u))*(float)q4b)*x[block*256+64+j];
        const int q5b=(int)((lo001b>>4)|(((hi01b>>4)&3)<<4))-32;
        sb5+=((db*(float)(signed char)((sc1b>>8)&255u))*(float)q5b)*x[block*256+80+j];
        const int q6b=(int)((lo010b>>4)|(((hi00b>>6)&3)<<4))-32;
        sb6+=((db*(float)(signed char)((sc1b>>16)&255u))*(float)q6b)*x[block*256+96+j];
        const int q7b=(int)((lo011b>>4)|(((hi01b>>6)&3)<<4))-32;
        sb7+=((db*(float)(signed char)((sc1b>>24)&255u))*(float)q7b)*x[block*256+112+j];
        const unsigned int hi10b=((c2b>>16)&255u);
        const unsigned int lo100b=((c1b>>0)&255u);
        const unsigned int lo110b=((c1b>>16)&255u);
        const unsigned int hi11b=((c2b>>24)&255u);
        const unsigned int lo101b=((c1b>>8)&255u);
        const unsigned int lo111b=((c1b>>24)&255u);
        const int q8b=(int)((lo100b&15)|(((hi10b>>0)&3)<<4))-32;
        sb8+=((db*(float)(signed char)((sc2b>>0)&255u))*(float)q8b)*x[block*256+128+j];
        const int q9b=(int)((lo101b&15)|(((hi11b>>0)&3)<<4))-32;
        sb9+=((db*(float)(signed char)((sc2b>>8)&255u))*(float)q9b)*x[block*256+144+j];
        const int q10b=(int)((lo110b&15)|(((hi10b>>2)&3)<<4))-32;
        sb10+=((db*(float)(signed char)((sc2b>>16)&255u))*(float)q10b)*x[block*256+160+j];
        const int q11b=(int)((lo111b&15)|(((hi11b>>2)&3)<<4))-32;
        sb11+=((db*(float)(signed char)((sc2b>>24)&255u))*(float)q11b)*x[block*256+176+j];
        const int q12b=(int)((lo100b>>4)|(((hi10b>>4)&3)<<4))-32;
        sb12+=((db*(float)(signed char)((sc3b>>0)&255u))*(float)q12b)*x[block*256+192+j];
        const int q13b=(int)((lo101b>>4)|(((hi11b>>4)&3)<<4))-32;
        sb13+=((db*(float)(signed char)((sc3b>>8)&255u))*(float)q13b)*x[block*256+208+j];
        const int q14b=(int)((lo110b>>4)|(((hi10b>>6)&3)<<4))-32;
        sb14+=((db*(float)(signed char)((sc3b>>16)&255u))*(float)q14b)*x[block*256+224+j];
        const int q15b=(int)((lo111b>>4)|(((hi11b>>6)&3)<<4))-32;
        sb15+=((db*(float)(signed char)((sc3b>>24)&255u))*(float)q15b)*x[block*256+240+j];
    
    }
    float suma=((((sa0+sa1)+(sa2+sa3))+((sa4+sa5)+(sa6+sa7)))+(((sa8+sa9)+(sa10+sa11))+((sa12+sa13)+(sa14+sa15))));
    float sumb=((((sb0+sb1)+(sb2+sb3))+((sb4+sb5)+(sb6+sb7)))+(((sb8+sb9)+(sb10+sb11))+((sb12+sb13)+(sb14+sb15))));
    for(int offset=16;offset>0;offset>>=1){suma+=__shfl_down_sync(0xffffffff,suma,offset);sumb+=__shfl_down_sync(0xffffffff,sumb,offset);}
    if(lane==0){raw[rowA]=suma;if(hasB)raw[rowB]=sumb;}
}
extern "C" __global__ void q6_matvec_raw_packed_dual(const unsigned char* c,const unsigned short* d,const float* x,float* raw,float* capped,unsigned int columns,unsigned int rows,float cap){q6_project_packed_dual(c,d,x,raw,columns,rows);}

// Batched head in two passes over each row (s0..s7, then s8..s15): the left
// half of the final tree is reduced after pass one, so only ~9 values per
// token stay live. Each accumulator still sees the same blocks in the same
// order and the tree is unchanged: bit-identical to q6_project_packed_batch.
template<int B> __device__ __forceinline__ void q6_project_packed_batch2(const unsigned char* codes,const unsigned short* dv,const float* x,float* raw,unsigned int columns,unsigned int rows) {
    const unsigned int lane=threadIdx.x&31,row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(row>=rows)return;
    const unsigned int j=lane&15,blocks=columns/256;
    float left[B];
    {
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
    }
    #pragma unroll
    for(int t=0;t<B;++t)left[t]=(((s0[t]+s1[t])+(s2[t]+s3[t]))+((s4[t]+s5[t])+(s6[t]+s7[t])));
    }
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
        float sum=left[t]+(((s8[t]+s9[t])+(s10[t]+s11[t]))+((s12[t]+s13[t])+(s14[t]+s15[t])));
        for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(lane==0)raw[(unsigned long long)t*rows+row]=sum;
    }
}
extern "C" __global__ void q6_matvec_raw_packed2_b1(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch2<1>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed2_b2(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch2<2>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed2_b3(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch2<3>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed2_b4(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch2<4>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed2_b5(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch2<5>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed2_b6(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch2<6>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed2_b7(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch2<7>(c,d,x,raw,columns,rows);}
extern "C" __global__ void q6_matvec_raw_packed2_b8(const unsigned char* c,const unsigned short* d,const float* x,float* raw,unsigned int columns,unsigned int rows){q6_project_packed_batch2<8>(c,d,x,raw,columns,rows);}
