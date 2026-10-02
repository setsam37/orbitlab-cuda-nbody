#include "orbitlab/cuda_simulator.hpp"
#include "device_buffer.hpp"
#include "forces_basic.cuh"
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
  if(kernel!=GpuKernel::Basic) throw std::logic_error("tiled not implemented");
  basic_force<R><<<(n+block-1)/block,block>>>(buffer.data,dest,n,softening_squared<R>(config));
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
template<class R> void GpuSimulator<R>::step(R) { throw std::logic_error("not implemented"); }
template<class R> void GpuSimulator<R>::reset(const State<R>& s) {
 validate_state(s);if(s.size()!=std::size_t(impl_->n)) throw std::invalid_argument("reset cannot change N");
 impl_->upload(s);force_only();synchronize();
}
template<class R> State<R> GpuSimulator<R>::download() {
 synchronize();State<R> s;std::vector<R>* fields[]={&s.mass,&s.x,&s.y,&s.z,&s.vx,&s.vy,&s.vz};
 for(int k=0;k<7;++k) { fields[k]->resize(impl_->n);CUDA_CHECK(cudaMemcpy(fields[k]->data(),impl_->buffer.data+std::size_t(k)*impl_->n,std::size_t(impl_->n)*sizeof(R),cudaMemcpyDeviceToHost)); }
 validate_state(s);return s;
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
template<class R> double GpuSimulator<R>::timed_batch(bool,int,R) { throw std::logic_error("not implemented"); }
template class GpuSimulator<float>;
template class GpuSimulator<double>;
}
