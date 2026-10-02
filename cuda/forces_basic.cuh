#pragma once
namespace orbitlab {
template<class R> __global__ void basic_force(const R* state,R* a,int n,R e2) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
 R xi=state[n+i],yi=state[2*n+i],zi=state[3*n+i],ax=0,ay=0,az=0;
 for(int j=0;j<n;++j) {
  if(i==j) continue;
  R dx=state[n+j]-xi,dy=state[2*n+j]-yi,dz=state[3*n+j]-zi;
  R r2=dx*dx+dy*dy+dz*dz+e2,scale=state[j]/(r2*sqrt(r2));
  ax+=dx*scale;ay+=dy*scale;az+=dz*scale;
 }
 a[i]=ax;a[n+i]=ay;a[2*n+i]=az;
}
}
