#include "gpu_test_support.hpp"
TEST_F(GpuTest, TimedForceAndStepHavePositiveDuration) {
 auto s=cloud<float>(17,37);GpuSimulator<float> gpu(s,{},GpuKernel::Tiled);
 EXPECT_GT(gpu.timed_batch(true,20,.001f),0);EXPECT_GT(gpu.timed_batch(false,20,.001f),0);EXPECT_NO_THROW(gpu.download());
 EXPECT_THROW(gpu.timed_batch(true,0,.001f),std::invalid_argument);
}
