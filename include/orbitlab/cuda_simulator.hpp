#pragma once
#include "physics.hpp"
#include <memory>
namespace orbitlab {
enum class GpuKernel { Basic,Tiled };
template<class R> class GpuSimulator {
 struct Impl;std::unique_ptr<Impl> impl_;
public:
 GpuSimulator(State<R>,ForceConfig,GpuKernel,int block_size=128);
 ~GpuSimulator();
 GpuSimulator(GpuSimulator&&) noexcept;
 GpuSimulator& operator=(GpuSimulator&&) noexcept;
 GpuSimulator(const GpuSimulator&)=delete;
 GpuSimulator& operator=(const GpuSimulator&)=delete;
 void step(R dt);
 void reset(const State<R>&);
 State<R> download();
 Acceleration<R> acceleration_snapshot();
 void force_only();
 void synchronize();
 double timed_batch(bool force_only_mode,int count,R dt);
};
extern template class GpuSimulator<float>;
extern template class GpuSimulator<double>;
}
