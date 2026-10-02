#pragma once
#include <gtest/gtest.h>
#include "orbitlab/cuda_simulator.hpp"
#include "orbitlab/comparison.hpp"
#include "cuda_check.hpp"
using namespace orbitlab;
class GpuTest:public testing::Test {
 void SetUp() override {
  int n=0;auto status=cudaGetDeviceCount(&n);
  if(status==cudaErrorNoDevice||status==cudaErrorInsufficientDriver||(status==cudaSuccess&&n==0)) GTEST_SKIP()<<"CUDA device unavailable";
  ASSERT_EQ(status,cudaSuccess)<<cudaGetErrorString(status);
 }
};
template<class To,class From> State<To> cast_state(const State<From>& s) {
 State<To> t;
 const std::vector<From>* src[]={&s.mass,&s.x,&s.y,&s.z,&s.vx,&s.vy,&s.vz};
 std::vector<To>* dst[]={&t.mass,&t.x,&t.y,&t.z,&t.vx,&t.vy,&t.vz};
 for(int k=0;k<7;++k) dst[k]->assign(src[k]->begin(),src[k]->end());return t;
}
template<class R> void force_checks(GpuKernel kernel) {
 int checks=0;
 for(int n:{1,2,17,63,64,65,127,128,129,257,1003}) for(int b:{64,128,256}) for(int variant=0;variant<3;++variant) {
  SCOPED_TRACE("N="+std::to_string(n)+" block="+std::to_string(b)+" variant="+std::to_string(variant));
  auto s=cloud<R>(n,37);
  if(variant==1) { for(int i=0;i<n;++i) { s.x[i]=0;s.y[i]=0;s.z[i]=0; } }
  if(variant==2&&n==2) { s=circular_pair<R>(.01);s.mass={2,3}; }
  auto expected=cpu_acceleration(cast_state<double>(s),{});GpuSimulator<R> sim(s,{},kernel,b);
  auto c=acceleration_matches(sim.acceleration_snapshot(),expected,sizeof(R)==4?1e-5:1e-11,sizeof(R)==4?3e-4:1e-10);
  EXPECT_TRUE(c.passed)<<"error="<<c.max_absolute<<" normalized="<<c.max_normalized;
  ++checks;
 }
 std::cout<<"force fixture configurations="<<checks<<" precision bytes="<<sizeof(R)<<"\n";
}
