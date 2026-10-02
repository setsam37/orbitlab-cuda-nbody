#include "gpu_test_support.hpp"
TEST_F(GpuTest, BasicFloatAllSizes) { force_checks<float>(GpuKernel::Basic); }
TEST_F(GpuTest, BasicDoubleAllSizes) { force_checks<double>(GpuKernel::Basic); }
TEST_F(GpuTest, RoundtripAndMovePreserveState) {
 auto s=cloud<double>(17,37);GpuSimulator<double> a(s,{},GpuKernel::Basic);auto b=std::move(a);auto t=b.download();
 EXPECT_EQ(s.mass,t.mass);EXPECT_EQ(s.x,t.x);EXPECT_EQ(s.vz,t.vz);
}
TEST_F(GpuTest, InvalidBlockSizeIsRejected) {
 EXPECT_THROW(GpuSimulator<double>(cloud<double>(2,37),{},GpuKernel::Basic,65),std::invalid_argument);
}
TEST(CudaErrors, ReportsCallAndLocation) {
 try { CUDA_CHECK(cudaErrorInvalidValue);FAIL()<<"Expected failure"; }
 catch(const std::runtime_error& e) { std::string message=e.what();EXPECT_NE(message.find("cudaErrorInvalidValue"),std::string::npos);EXPECT_NE(message.find("cuda_force_test.cpp"),std::string::npos); }
}
