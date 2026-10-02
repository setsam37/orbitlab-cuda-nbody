#pragma once
namespace orbitlab {
template<class R> __global__ void update_positions(R* s,const R* a,int n,R dt) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
 for(int k=0;k<3;++k) s[(k+1)*n+i]+=s[(k+4)*n+i]*dt+R(.5)*a[k*n+i]*dt*dt;
}
template<class R> __global__ void update_velocities(R* s,const R* olda,const R* newa,int n,R dt) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
 for(int k=0;k<3;++k) s[(k+4)*n+i]+=R(.5)*(olda[k*n+i]+newa[k*n+i])*dt;
}
}
