#include <gtest/gtest.h>
#include "orbitlab/fixtures.hpp"
using namespace orbitlab;
State<double> one() { return {{1},{0},{0},{0},{1},{2},{3}}; }
TEST(State, AcceptsFinitePositiveBody) { EXPECT_NO_THROW(validate_state(one())); }
TEST(State, RejectsEmpty) { EXPECT_THROW(validate_state(State<double>{}),std::invalid_argument); }
TEST(State, RejectsUnequalLengths) { auto s=one(); s.y.clear(); EXPECT_THROW(validate_state(s),std::invalid_argument); }
TEST(State, RejectsNonpositiveMass) { for(double m:{0.,-1.}) { auto s=one();s.mass[0]=m;EXPECT_THROW(validate_state(s),std::invalid_argument); } }
TEST(State, RejectsNonfiniteComponents) {
 for(double v:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
  for(int k=0;k<7;++k) { auto s=one();std::vector<double>* fields[]={&s.mass,&s.x,&s.y,&s.z,&s.vx,&s.vy,&s.vz};(*fields[k])[0]=v;EXPECT_THROW(validate_state(s),std::invalid_argument); }
 }
}
TEST(State, RejectsCountOverflow) { EXPECT_THROW(validate_count<double>(std::numeric_limits<std::size_t>::max()),std::invalid_argument); }
TEST(Fixture, CircularPairUsesSoftenedSpeed) {
 auto s=circular_pair<double>(.01);ASSERT_EQ(s.size(),2u);
 EXPECT_DOUBLE_EQ(s.x[0],-.5);EXPECT_DOUBLE_EQ(s.x[1],.5);
 EXPECT_DOUBLE_EQ(s.mass[0],1);EXPECT_DOUBLE_EQ(s.mass[1],1);
 EXPECT_NEAR(s.vy[1],std::sqrt(.5/std::pow(1.0001,1.5)),1e-15);
 EXPECT_DOUBLE_EQ(s.vy[0],-s.vy[1]);EXPECT_NO_THROW(validate_state(s));
 EXPECT_NEAR(circular_period(.01),2*std::acos(-1.)/(2*s.vy[1]),1e-14);
}
TEST(Fixture, SeededCloudIsBoundedAndRepeatable) {
 auto a=cloud<double>(17,37),b=cloud<double>(17,37),c=cloud<double>(17,38);
 EXPECT_EQ(a.x,b.x);EXPECT_EQ(a.vz,b.vz);EXPECT_NE(a.x,c.x);EXPECT_NO_THROW(validate_state(a));
 for(std::size_t i=0;i<a.size();++i) { EXPECT_GT(a.mass[i],0);EXPECT_LE(std::abs(a.x[i]),1);EXPECT_LE(std::abs(a.y[i]),1);EXPECT_LE(std::abs(a.z[i]),1);EXPECT_LE(std::abs(a.vx[i]),.1); }
}
TEST(Fixture, RejectsInvalidSofteningAndCount) {
 EXPECT_THROW(circular_pair<double>(0),std::invalid_argument);
 EXPECT_THROW(cloud<double>(0,37),std::invalid_argument);
}
