// Native ARM correctness-preserving dense Conv baseline. No device accesses.
// INT16 input-minus-zero-point, INT16 weights, INT32 multiply/accumulate.
// Logical product order is unchanged; callers verify accumulator range.
static int8_t rs1_rq(int32_t v,uint32_t m,uint32_t s,int zp){
 int64_t z=int64_t(v)*m;bool neg=z<0;uint64_t a=neg?-z:z,q=a>>s,r=a&((uint64_t(1)<<s)-1),half=uint64_t(1)<<(s?s-1:0);
 q+=s&&(r>half||(r==half&&(q&1)));int64_t out=(neg?-int64_t(q):int64_t(q))+zp;
 return int8_t(std::max(int64_t(-128),std::min(int64_t(127),out)));
}
static void rs1_dense(const int8_t*x,const int8_t*w,const int32_t*b,
 const uint32_t*mult,const uint32_t*shift,const uint8_t*lut,int8_t*out,int32_t*raw,
 int ci,int h,int width,int co,int oh,int ow,int k,int stride,int pad,int xzp,int wzp,int ozp,int mode){
 if(ci<=0||ci>512||co<=0||h<=0||width<=0||oh<=0||ow<=0||(k!=1&&k!=3))
  throw std::runtime_error("native CPU dense shape contract");
 const int K=ci*k*k,N=oh*ow,B=64;
 int64_t xmax=std::max(std::abs(-128-xzp),std::abs(127-xzp));
 for(int c=0;c<co;c++){int64_t bound=std::abs(int64_t(b[c]));
  for(int z=0;z<K;z++)bound+=xmax*std::abs(int(w[size_t(c)*K+z])-wzp);
  if(bound>2147483647ll)throw std::runtime_error("native CPU accumulator bound");
 }

 std::vector<int16_t> patch(size_t(K)*B);
 for(int first=0;first<N;first+=B){
  int count=std::min(B,N-first);
  for(int c=0;c<ci;c++)for(int ky=0;ky<k;ky++)for(int kx=0;kx<k;kx++){
   int z=(c*k+ky)*k+kx;int16_t*dst=patch.data()+size_t(z)*count;
   for(int j=0;j<count;j++){int pos=first+j,sy=pos/ow*stride-pad+ky,sx=pos%ow*stride-pad+kx;
    dst[j]=sy<0||sy>=h||sx<0||sx>=width?0:int(x[(c*h+sy)*width+sx])-xzp;
   }
  }
  #pragma omp parallel for schedule(static) num_threads(2) if(co>=16&&K*count>=512)
  for(int oc=0;oc<co;oc+=4){
   int nc=std::min(4,co-oc),j=0;
   auto save=[&](int c,int p,int32_t value){size_t at=size_t(c)*N+first+p;
    if(raw)raw[at]=value;
    else {int8_t q=rs1_rq(value,mult[c],shift[c],ozp);out[at]=mode==2?int8_t(lut[uint8_t(q)]):q;}
   };
#ifdef __ARM_NEON
   if(nc==4)for(;j+8<=count;j+=8){
    int32x4_t a0=vdupq_n_s32(b[oc]),a1=a0,a2=vdupq_n_s32(b[oc+1]),a3=a2;
    int32x4_t a4=vdupq_n_s32(b[oc+2]),a5=a4,a6=vdupq_n_s32(b[oc+3]),a7=a6;
    for(int z=0;z<K;z++){
     int16x8_t xx=vld1q_s16(patch.data()+size_t(z)*count+j);
     int16x4_t lo=vget_low_s16(xx),hi=vget_high_s16(xx);
     int16_t w0=int(w[size_t(oc)*K+z])-wzp,w1=int(w[size_t(oc+1)*K+z])-wzp;
     int16_t w2=int(w[size_t(oc+2)*K+z])-wzp,w3=int(w[size_t(oc+3)*K+z])-wzp;
     a0=vmlal_n_s16(a0,lo,w0);a1=vmlal_n_s16(a1,hi,w0);
     a2=vmlal_n_s16(a2,lo,w1);a3=vmlal_n_s16(a3,hi,w1);
     a4=vmlal_n_s16(a4,lo,w2);a5=vmlal_n_s16(a5,hi,w2);
     a6=vmlal_n_s16(a6,lo,w3);a7=vmlal_n_s16(a7,hi,w3);
    }
    int32_t acc[4][8];vst1q_s32(acc[0],a0);vst1q_s32(acc[0]+4,a1);
    vst1q_s32(acc[1],a2);vst1q_s32(acc[1]+4,a3);vst1q_s32(acc[2],a4);vst1q_s32(acc[2]+4,a5);
    vst1q_s32(acc[3],a6);vst1q_s32(acc[3]+4,a7);
    for(int c=0;c<4;c++)for(int l=0;l<8;l++)save(oc+c,j+l,acc[c][l]);
   }
#endif
   for(;j<count;j++)for(int c=0;c<nc;c++){
    int32_t acc=b[oc+c];for(int z=0;z<K;z++)acc+=int32_t(patch[size_t(z)*count+j])*(int(w[size_t(oc+c)*K+z])-wzp);
    save(oc+c,j,acc);
   }
  }
 }
}
