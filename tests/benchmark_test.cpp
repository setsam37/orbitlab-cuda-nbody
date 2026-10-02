#include <gtest/gtest.h>
#include "orbitlab/cpu_simulator.hpp"
using namespace orbitlab;
TEST(Benchmark, ReusedForceOutputResizesAndClears) {
 auto s=circular_pair<double>(.01);Acceleration<double> a{{999},{999},{999}};cpu_acceleration_into(s,{},a);
 ASSERT_EQ(a.x.size(),2u);EXPECT_NEAR(a.x[0],1/std::pow(1.0001,1.5),1e-12);
 auto one=cloud<double>(1,37);cpu_acceleration_into(one,{},a);ASSERT_EQ(a.x.size(),1u);EXPECT_DOUBLE_EQ(a.x[0],0);EXPECT_DOUBLE_EQ(a.y[0],0);
}
TEST(Benchmark, ForceOnlyPreservesParticleState) {
 auto s=cloud<double>(17,37);CpuSimulator<double> sim(s,{});EXPECT_NO_THROW(sim.force_only());EXPECT_EQ(sim.state().x,s.x);EXPECT_EQ(sim.state().vx,s.vx);
}
