#include "orbitlab/cuda_simulator.hpp"
#include "device_buffer.hpp"
#include "forces_basic.cuh"
#include "forces_tiled.cuh"
#include "integrate.cuh"
#include "cuda_event.hpp"
namespace orbitlab {
template<class R> struct GpuSimulator<R>::Impl {
 int n,block;ForceConfig config;GpuKernel kernel;DeviceBuffer<R> buffer;
 R* olda;R* newa;
 Impl(const State<R>& s,ForceConfig c,GpuKernel k,int b):n(int(s.size())),block(b),config(c),kernel(k),buffer(13*s.size()),olda(buffer.data+7*s.size()),newa(buffer.data+10*s.size()) {}
 void upload(const State<R>& s) {
  const std::vector<R>* fields[]={&s.mass,&s.x,&s.y,&s.z,&s.vx,&s.vy,&s.vz};
  for(int k=0;k<7;++k) CUDA_CHECK(cudaMemcpy(buffer.data+std::size_t(k)*n,fields[k]->data(),std::size_t(n)*sizeof(R),cudaMemcpyHostToDevice));
 }
 void force(R* dest) {
  int grid=(n+block-1)/block;R e2=softening_squared<R>(config);
  if(kernel==GpuKernel::Basic) basic_force<R><<<grid,block>>>(buffer.data,dest,n,e2);
  else if(block==64) tiled_force<R,64><<<grid,64>>>(buffer.data,dest,n,e2);
  else if(block==128) tiled_force<R,128><<<grid,128>>>(buffer.data,dest,n,e2);
  else tiled_force<R,256><<<grid,256>>>(buffer.data,dest,n,e2);
  CUDA_CHECK(cudaGetLastError());
 }
};
template<class R> GpuSimulator<R>::GpuSimulator(State<R> s,ForceConfig c,GpuKernel k,int block) {
 validate_state(s);softening_squared<R>(c);
 if(block!=64&&block!=128&&block!=256) throw std::invalid_argument("block size must be 64, 128, or 256");
 impl_=std::make_unique<Impl>(s,c,k,block);reset(s);
}
template<class R> GpuSimulator<R>::~GpuSimulator()=default;
template<class R> GpuSimulator<R>::GpuSimulator(GpuSimulator&&) noexcept=default;
template<class R> GpuSimulator<R>& GpuSimulator<R>::operator=(GpuSimulator&&) noexcept=default;
template<class R> void GpuSimulator<R>::step(R dt) {
 if(!std::isfinite(dt)||dt<=0) throw std::invalid_argument("dt must be finite and positive");
 int n=impl_->n,b=impl_->block,grid=(n+b-1)/b;
 update_positions<R><<<grid,b>>>(impl_->buffer.data,impl_->olda,n,dt);CUDA_CHECK(cudaGetLastError());
 impl_->force(impl_->newa);
 update_velocities<R><<<grid,b>>>(impl_->buffer.data,impl_->olda,impl_->newa,n,dt);CUDA_CHECK(cudaGetLastError());
 std::swap(impl_->olda,impl_->newa);
}
template<class R> void GpuSimulator<R>::reset(const State<R>& s) {
 validate_state(s);if(s.size()!=std::size_t(impl_->n)) throw std::invalid_argument("reset cannot change N");
 impl_->upload(s);force_only();synchronize();
}
template<class R> State<R> GpuSimulator<R>::download() {
 State<R> s;download_into(s);return s;
}
template<class R> void GpuSimulator<R>::download_into(State<R>& s) {
 synchronize();std::vector<R>* fields[]={&s.mass,&s.x,&s.y,&s.z,&s.vx,&s.vy,&s.vz};
 for(int k=0;k<7;++k) { fields[k]->resize(impl_->n);CUDA_CHECK(cudaMemcpy(fields[k]->data(),impl_->buffer.data+std::size_t(k)*impl_->n,std::size_t(impl_->n)*sizeof(R),cudaMemcpyDeviceToHost)); }
 validate_state(s);
}
template<class R> Acceleration<R> GpuSimulator<R>::acceleration_snapshot() {
 synchronize();Acceleration<R> a;std::vector<R>* fields[]={&a.x,&a.y,&a.z};
 for(int k=0;k<3;++k) {
  fields[k]->resize(impl_->n);CUDA_CHECK(cudaMemcpy(fields[k]->data(),impl_->olda+std::size_t(k)*impl_->n,std::size_t(impl_->n)*sizeof(R),cudaMemcpyDeviceToHost));
  for(R v:*fields[k]) if(!std::isfinite(v)) throw std::runtime_error("nonfinite GPU acceleration");
 }
 return a;
}
template<class R> void GpuSimulator<R>::force_only() { impl_->force(impl_->olda); }
template<class R> void GpuSimulator<R>::synchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }
template<class R> double GpuSimulator<R>::timed_batch(bool force_mode,int count,R dt) {
 if(count<=0||count>10000) throw std::invalid_argument("invalid timing repetition count");
 if(!std::isfinite(dt)||dt<=0) throw std::invalid_argument("invalid timing dt");
 CudaEvent begin,end;CUDA_CHECK(cudaEventRecord(begin.value,0));
 for(int k=0;k<count;++k) { if(force_mode) force_only();else step(dt); }
 CUDA_CHECK(cudaEventRecord(end.value,0));CUDA_CHECK(cudaEventSynchronize(end.value));
 float elapsed=0;CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin.value,end.value));
 if(!std::isfinite(elapsed)||elapsed<=0) throw std::runtime_error("invalid CUDA event duration");
 return double(elapsed)/count;
}
template class GpuSimulator<float>;
template class GpuSimulator<double>;
}
