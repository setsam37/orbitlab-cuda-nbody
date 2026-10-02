#include <gtest/gtest.h>
#include "orbitlab/io.hpp"
using namespace orbitlab;
TEST(IO, RoundtripRetainsDoublePrecision) {
 auto s=cloud<double>(17,37);std::stringstream out;write_state(out,s);auto t=read_state<double>(out);
 EXPECT_EQ(s.x,t.x);EXPECT_EQ(s.mass,t.mass);EXPECT_EQ(s.vz,t.vz);
}
TEST(IO, ReadsHandWrittenFixture) {
 std::stringstream in("id,mass,x,y,z,vx,vy,vz\n0,2,1,0,0,0,3,0\n");auto s=read_state<double>(in);
 ASSERT_EQ(s.size(),1u);EXPECT_DOUBLE_EQ(s.mass[0],2);EXPECT_DOUBLE_EQ(s.vy[0],3);
}
TEST(IO, RejectsMalformedInput) {
 for(auto row:{"0,1,0,0,0,0,0", "0,1,0,0,0,0,0,NaN", "1,1,0,0,0,0,0,0", "0,-1,0,0,0,0,0,0", "0,1oops,0,0,0,0,0,0"}) {
  std::stringstream in(std::string("id,mass,x,y,z,vx,vy,vz\n")+row+"\n");EXPECT_THROW(read_state<double>(in),std::invalid_argument);
 }
 std::stringstream duplicate("id,mass,x,y,z,vx,vy,vz\n0,1,0,0,0,0,0,0\n0,1,0,0,0,0,0,0\n");
 EXPECT_THROW(read_state<double>(duplicate),std::invalid_argument);
}
TEST(IO, DetectsFailedOutputStream) {
 std::ostringstream out;out.setstate(std::ios::badbit);EXPECT_THROW(write_state(out,circular_pair<double>(.01)),std::runtime_error);
}
TEST(IO, FramesUseProvidedStepAndTime) {
 State<double> s{{2},{1},{0},{0},{0},{3},{0}};std::ostringstream out;write_frame(out,7,.25,s);
 EXPECT_EQ(out.str(),"7,0.25,0,1,0,0,0,3,0\n");
}
