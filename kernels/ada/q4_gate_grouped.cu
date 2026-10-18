// Launch contract: blockDim.x = 128 * Groups; columns <= 3840 for Stage.
// Research only: independent four-row groups share a CTA, optionally staging x.
template<int Groups, bool Stage> __device__ void q4_group_body(const unsigned short* codes,const unsigned short* scales,const float* x,float* y,unsigned columns,unsigned rows){
 const unsigned local=threadIdx.x%128, group=threadIdx.x/128, lane=local%32;
 const unsigned row=(blockIdx.x*Groups+group)*4+(lane&3), j=(lane>>2)*2, blocks=columns/32;
 __shared__ float staged[Stage?3840:1];
 __shared__ float partial[128*Groups];
 if constexpr(Stage){for(unsigned i=threadIdx.x;i<columns;i+=128*Groups)staged[i]=x[i]; __syncthreads();}
 const float* input=Stage?staged:x;
 float a=0,b=0,c=0,d=0;
 if((row&~3u)<rows){
 for(unsigned k=local/32;k<blocks;k+=4){
  const unsigned long long tile=(unsigned long long)(row/4)*blocks+k;
  const float scale=decode_half(scales[tile*4+(lane&3)]);
  const unsigned q=codes[(tile*8+(lane>>2))*4+(lane&3)],lo=q&255,hi=q>>8;
  a+=(scale*(float)((int)(lo&15)-8))*input[k*32+j];
  b+=(scale*(float)((int)(hi&15)-8))*input[k*32+j+1];
  c+=(scale*(float)((int)(lo>>4)-8))*input[k*32+j+16];
  d+=(scale*(float)((int)(hi>>4)-8))*input[k*32+j+17];
 }}
 partial[threadIdx.x]=(a+b)+(c+d); __syncthreads();
 if(local<32){
  const float* p=partial+group*128;
  float sum=(p[lane]+p[64+lane])+(p[32+lane]+p[96+lane]);
  for(int offset=16;offset>=4;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
  if(lane<4 && row<rows)y[row]=sum;
 }
}
extern "C" __global__ void q4_group_g2(const unsigned short* c,const unsigned short* s,const float* x,float* y,unsigned cols,unsigned rows){q4_group_body<2,false>(c,s,x,y,cols,rows);}
extern "C" __global__ void q4_group_g4(const unsigned short* c,const unsigned short* s,const float* x,float* y,unsigned cols,unsigned rows){q4_group_body<4,false>(c,s,x,y,cols,rows);}
extern "C" __global__ void q4_group_s2(const unsigned short* c,const unsigned short* s,const float* x,float* y,unsigned cols,unsigned rows){q4_group_body<2,true>(c,s,x,y,cols,rows);}
extern "C" __global__ void q4_group_s4(const unsigned short* c,const unsigned short* s,const float* x,float* y,unsigned cols,unsigned rows){q4_group_body<4,true>(c,s,x,y,cols,rows);}
