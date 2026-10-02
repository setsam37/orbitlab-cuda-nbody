#include <gtest/gtest.h>
#include "orbitlab/cpu_simulator.hpp"
#include "orbitlab/diagnostics.hpp"
using namespace orbitlab;
TEST(Integrator, IsolatedMotionIsLinear) {
 State<double> s{{1},{0},{0},{0},{1},{2},{-3}};
 for(auto m:{Integrator::Verlet,Integrator::Euler}) {
  CpuSimulator<double> sim(s,{},m);for(int k=0;k<10;++k) sim.step(.1);
  EXPECT_NEAR(sim.state().x[0],1,1e-14);EXPECT_NEAR(sim.state().y[0],2,1e-14);EXPECT_NEAR(sim.state().z[0],-3,1e-14);EXPECT_DOUBLE_EQ(sim.state().vx[0],1);
 }
}
TEST(Integrator, OneVerletStepUsesBothAccelerations) {
 State<double> s{{1,1},{-.5,.5},{0,0},{0,0},{0,0},{0,0},{0,0}};
 CpuSimulator<double> sim(s,{1});sim.step(.1);
 double a0=1/std::pow(2.,1.5),x=-.5+.005*a0,sep=-2*x;
 double a1=sep/std::pow(sep*sep+1,1.5);
 EXPECT_NEAR(sim.state().x[0],x,1e-15);EXPECT_NEAR(sim.state().vx[0],.05*(a0+a1),1e-15);
}
TEST(Integrator, EulerUsesOldStateForBothUpdates) {
 State<double> s{{1,1},{-.5,.5},{0,0},{0,0},{1,-1},{0,0},{0,0}};
 CpuSimulator<double> sim(s,{1},Integrator::Euler);sim.step(.1);
 EXPECT_NEAR(sim.state().x[0],-.4,1e-15);EXPECT_NEAR(sim.state().vx[0],1+.1/std::pow(2.,1.5),1e-15);
}
TEST(Integrator, ResetRecomputesCachedAcceleration) {
 auto s=circular_pair<double>(.01);CpuSimulator<double> reused(s,{});reused.step(.01);
 s.x={-1,1};CpuSimulator<double> fresh(s,{});reused.reset(s);reused.step(.01);fresh.step(.01);
 EXPECT_EQ(reused.state().x,fresh.state().x);EXPECT_EQ(reused.state().vx,fresh.state().vx);
}
TEST(Integrator, RejectsInvalidStep) {
 CpuSimulator<double> sim(circular_pair<double>(.01),{});
 EXPECT_THROW(sim.step(0),std::invalid_argument);
 EXPECT_THROW(sim.step(-1),std::invalid_argument);
 EXPECT_THROW(sim.step(std::numeric_limits<double>::quiet_NaN()),std::invalid_argument);
}
double orbit_error(int steps) {
 auto s=circular_pair<double>(.01);CpuSimulator<double> sim(s,{});double dt=circular_period(.01)/steps;
 for(int k=0;k<steps;++k) sim.step(dt);
 auto t=sim.state();double e=0;
 for(int i=0;i<2;++i) for(double v:{t.x[i]-s.x[i],t.y[i],t.z[i],t.vx[i],t.vy[i]-s.vy[i],t.vz[i]}) e+=v*v;
 return std::sqrt(e);
}
TEST(Convergence, VerletIsSecondOrderForSoftenedCircularOrbit) {
 double a=orbit_error(128),b=orbit_error(256),c=orbit_error(512);
 std::cout<<"convergence errors="<<a<<","<<b<<","<<c<<" ratios="<<a/b<<","<<b/c<<"\n";
 EXPECT_GE(a/b,3.5);EXPECT_LE(a/b,4.5);EXPECT_GE(b/c,3.5);EXPECT_LE(b/c,4.5);
}
template<class R> void check_invariants() {
 auto s=circular_pair<R>(.01);CpuSimulator<R> sim(s,{});auto initial=diagnostics(s,{});
 double maxe=0,maxp=0,maxl=0;R dt=R(circular_period(.01)/512);
 for(int k=0;k<5120;++k) {
  sim.step(dt);auto d=diagnostics(sim.state(),{});
  maxe=std::max(maxe,std::abs(d.energy-initial.energy)/std::abs(initial.energy));
  maxp=std::max(maxp,std::hypot(d.momentum[0],d.momentum[1],d.momentum[2])/(2*std::abs(double(s.vy[0]))));
  maxl=std::max(maxl,std::abs(d.angular_momentum[2]-initial.angular_momentum[2])/std::abs(initial.angular_momentum[2]));
 }
 std::cout<<"invariants bytes="<<sizeof(R)<<" max energy="<<maxe<<" momentum="<<maxp<<" angular="<<maxl<<"\n";
 EXPECT_LE(maxe,1e-3);EXPECT_LE(maxp,1e-5);EXPECT_LE(maxl,1e-3);
}
TEST(Invariants, FloatCircularOrbit) { check_invariants<float>(); }
TEST(Invariants, DoubleCircularOrbit) { check_invariants<double>(); }
