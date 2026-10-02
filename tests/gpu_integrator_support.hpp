#pragma once
#include "gpu_test_support.hpp"
#include "orbitlab/cpu_simulator.hpp"
#include "orbitlab/diagnostics.hpp"
template<class R> void short_trajectory(GpuKernel kernel) {
 for(int n:{1,2,17,65,257}) {
  auto s=cloud<R>(n,37);GpuSimulator<R> gpu(s,{},kernel);CpuSimulator<double> cpu(cast_state<double>(s),{});
  for(int k=0;k<10;++k) {
   gpu.step(R(.001));cpu.step(double(R(.001)));auto g=gpu.download();auto c=cpu.state();
   auto p=acceleration_matches(Acceleration<R>{g.x,g.y,g.z},Acceleration<double>{c.x,c.y,c.z},sizeof(R)==4?1e-5:1e-10,sizeof(R)==4?1e-3:1e-9);
   auto v=acceleration_matches(Acceleration<R>{g.vx,g.vy,g.vz},Acceleration<double>{c.vx,c.vy,c.vz},sizeof(R)==4?1e-5:1e-10,sizeof(R)==4?1e-3:1e-9);
   EXPECT_TRUE(p.passed)<<"N="<<n<<" step="<<k+1<<" position "<<p.max_normalized;
   EXPECT_TRUE(v.passed)<<"N="<<n<<" step="<<k+1<<" velocity "<<v.max_normalized;
  }
 }
}
template<class R> void gpu_invariants(GpuKernel kernel) {
 auto s=circular_pair<R>(.01);GpuSimulator<R> gpu(s,{},kernel);auto initial=diagnostics(s,{});
 double e=0,p=0,l=0;
 for(int k=0;k<5120;++k) {
  gpu.step(R(circular_period(.01)/512));auto d=diagnostics(gpu.download(),{});
  e=std::max(e,std::abs(d.energy-initial.energy)/std::abs(initial.energy));
  p=std::max(p,std::hypot(d.momentum[0],d.momentum[1],d.momentum[2])/(2*std::abs(double(s.vy[0]))));
  l=std::max(l,std::abs(d.angular_momentum[2]-initial.angular_momentum[2])/std::abs(initial.angular_momentum[2]));
 }
 std::cout<<"GPU invariants precision="<<sizeof(R)<<" kernel="<<int(kernel)<<" energy="<<e<<" momentum="<<p<<" angular="<<l<<"\n";
 EXPECT_LE(e,1e-3);EXPECT_LE(p,1e-5);EXPECT_LE(l,1e-3);
}
