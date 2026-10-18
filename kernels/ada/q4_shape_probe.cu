// Research schedules: preserve the native q4_row accumulation/reduction tree.
// Four K partitions live in four warps; 1/2/4 rows share each CTA.
template<int Rows> __device__ void q4_shape_body(const unsigned char* w,
 const float* x,float* y,unsigned int columns,unsigned int rows){
    __shared__ float partial[128];
    const unsigned int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const unsigned int row=blockIdx.x*Rows+lane/8,j=(lane&7)*2,blocks=columns/32;
    float a=0,b=0,c=0,d=0;
    if(lane<Rows*8 && row<rows){
        for(unsigned int k=warp;k<blocks;k+=4){
            const unsigned char* p=w+((unsigned long long)row*blocks+k)*18;
            const float scale=decode_half(*(const unsigned short*)p);
            const unsigned int q=*(const unsigned short*)(p+2+j),lo=q&255,hi=q>>8;
            a+=(scale*(float)((int)(lo&15)-8))*x[k*32+j];
            b+=(scale*(float)((int)(hi&15)-8))*x[k*32+j+1];
            c+=(scale*(float)((int)(lo>>4)-8))*x[k*32+j+16];
            d+=(scale*(float)((int)(hi>>4)-8))*x[k*32+j+17];
        }
    }
    partial[threadIdx.x]=(a+b)+(c+d);
    __syncthreads();
    if(warp==0){
        float sum=(partial[lane]+partial[64+lane])+(partial[32+lane]+partial[96+lane]);
        for(int offset=4;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset,8);
        if(lane<Rows*8 && (lane&7)==0 && row<rows)y[row]=sum;
    }
}
extern "C" __global__ void q4_shape1(const unsigned char* w,const float* x,float* y,unsigned int columns,unsigned int rows){q4_shape_body<1>(w,x,y,columns,rows);}
extern "C" __global__ void q4_shape2(const unsigned char* w,const float* x,float* y,unsigned int columns,unsigned int rows){q4_shape_body<2>(w,x,y,columns,rows);}
extern "C" __global__ void q4_shape4(const unsigned char* w,const float* x,float* y,unsigned int columns,unsigned int rows){q4_shape_body<4>(w,x,y,columns,rows);}
