#include <gtest/gtest.h>
#include "orbitlab/physics.hpp"
#include "orbitlab/diagnostics.hpp"
#include "orbitlab/comparison.hpp"
using namespace orbitlab;
TEST(Physics, IsolatedBodyHasZeroAcceleration) {
 auto s=cloud<double>(1,37);auto a=cpu_acceleration(s,{});
 EXPECT_DOUBLE_EQ(a.x[0],0);EXPECT_DOUBLE_EQ(a.y[0],0);EXPECT_DOUBLE_EQ(a.z[0],0);
}
TEST(Physics, PairHasCorrectMagnitudeAndOppositeForces) {
 auto s=circular_pair<double>(.01);s.mass={2,3};auto a=cpu_acceleration(s,{});
 EXPECT_NEAR(a.x[0],3/std::pow(1.0001,1.5),1e-12);
 EXPECT_NEAR(a.x[1],-2/std::pow(1.0001,1.5),1e-12);
 EXPECT_NEAR(2*a.x[0]+3*a.x[1],0,1e-12);
}
TEST(Physics, CoincidentBodiesRemainFinite) {
 auto s=circular_pair<double>(.01);s.x={0,0};s.vy={0,0};auto a=cpu_acceleration(s,{});
 EXPECT_DOUBLE_EQ(a.x[0],0);EXPECT_DOUBLE_EQ(a.x[1],0);
 EXPECT_NEAR(diagnostics(s,{}).energy,-100,1e-12);
}
TEST(Physics, RejectsInvalidSoftening) {
 auto s=circular_pair<double>(.01);
 EXPECT_THROW(cpu_acceleration(s,{0}),std::invalid_argument);
 EXPECT_THROW(cpu_acceleration(s,{-1}),std::invalid_argument);
 EXPECT_THROW(cpu_acceleration(s,{std::numeric_limits<double>::infinity()}),std::invalid_argument);
}
TEST(Diagnostics, ComputesKineticMomentumAndAngularMomentum) {
 State<double> s{{2},{1},{0},{0},{0},{3},{0}};auto d=diagnostics(s,{});
 EXPECT_DOUBLE_EQ(d.energy,9);EXPECT_DOUBLE_EQ(d.momentum[1],6);EXPECT_DOUBLE_EQ(d.angular_momentum[2],6);
}
TEST(Diagnostics, CountsEachPotentialPairOnce) {
 State<double> s{{2,3},{0,1},{0,0},{0,0},{0,0},{0,0},{0,0}};
 EXPECT_NEAR(diagnostics(s,{}).energy,-6/std::sqrt(1.0001),1e-12);
}
TEST(Comparison, AbsoluteFloorHandlesZeroReference) {
 Acceleration<double> expected{{0},{0},{0}},near{{1e-6},{0},{0}},far{{1e-3},{0},{0}};
 EXPECT_TRUE(acceleration_matches(near,expected,1e-5,3e-4).passed);
 EXPECT_FALSE(acceleration_matches(far,expected,1e-5,3e-4).passed);
}
TEST(Comparison, VectorNormAndRelativeScale) {
 Acceleration<double> expected{{3},{4},{0}},near{{3.0001},{4},{0}};
 auto c=acceleration_matches(near,expected,1e-5,3e-4);EXPECT_TRUE(c.passed);EXPECT_NEAR(c.max_absolute,1e-4,1e-14);
 Acceleration<double> too_far{{3.001},{4.001},{.001}};
 EXPECT_FALSE(acceleration_matches(too_far,expected,1e-5,3e-4).passed);
}
TEST(Comparison, RejectsNonfiniteAndMalformedVectors) {
 Acceleration<double> ref{{0},{0},{0}},bad{{std::numeric_limits<double>::quiet_NaN()},{0},{0}};
 EXPECT_FALSE(acceleration_matches(bad,ref,1e-5,3e-4).passed);
 bad.x.clear();EXPECT_FALSE(acceleration_matches(bad,ref,1e-5,3e-4).passed);
}
