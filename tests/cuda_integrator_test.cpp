#include "gpu_integrator_support.hpp"
TEST_F(GpuTest, BasicFloatShortTrajectory) { short_trajectory<float>(GpuKernel::Basic); }
TEST_F(GpuTest, BasicDoubleShortTrajectory) { short_trajectory<double>(GpuKernel::Basic); }
TEST_F(GpuTest, BasicFloatInvariants) { gpu_invariants<float>(GpuKernel::Basic); }
TEST_F(GpuTest, BasicDoubleInvariants) { gpu_invariants<double>(GpuKernel::Basic); }
TEST_F(GpuTest, ResetRecomputesAcceleration) {
 auto s=circular_pair<double>(.01);GpuSimulator<double> sim(s,{},GpuKernel::Basic);sim.step(.01);
 s.x={-1,1};sim.reset(s);CpuSimulator<double> cpu(s,{});sim.step(.01);cpu.step(.01);
 EXPECT_NEAR(sim.download().vx[0],cpu.state().vx[0],1e-12);
 EXPECT_THROW(sim.step(0),std::invalid_argument);
}
