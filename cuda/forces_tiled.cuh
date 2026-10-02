#pragma once
namespace orbitlab {
template<class R,int B> __global__ void tiled_force(const R* state,R* a,int n,R e2) {
 __shared__ R tile[4][B];
 int lane=threadIdx.x,i=blockIdx.x*B+lane;bool valid=i<n;
 R xi=valid?state[n+i]:0,yi=valid?state[2*n+i]:0,zi=valid?state[3*n+i]:0;
 R ax=0,ay=0,az=0;
 for(int base=0;base<n;base+=B) {
  int source=base+lane;
  for(int k=0;k<4;++k) tile[k][lane]=source<n?state[k*n+source]:R(0);
  __syncthreads();
  int count=min(B,n-base);
  if(valid) for(int j=0;j<count;++j) {
   if(base+j==i) continue;
   R dx=tile[1][j]-xi,dy=tile[2][j]-yi,dz=tile[3][j]-zi;
   R r2=dx*dx+dy*dy+dz*dz+e2,scale=tile[0][j]/(r2*sqrt(r2));
   ax+=dx*scale;ay+=dy*scale;az+=dz*scale;
  }
  __syncthreads();
 }
 if(valid) { a[i]=ax;a[n+i]=ay;a[2*n+i]=az; }
}
}
