#include "gpu_integrator_support.hpp"
TEST_F(GpuTest, TiledFloatPartialTileAllSizes) { force_checks<float>(GpuKernel::Tiled); }
TEST_F(GpuTest, TiledDoublePartialTileAllSizes) { force_checks<double>(GpuKernel::Tiled); }
TEST_F(GpuTest, TiledFloatShortTrajectory) { short_trajectory<float>(GpuKernel::Tiled); }
TEST_F(GpuTest, TiledDoubleShortTrajectory) { short_trajectory<double>(GpuKernel::Tiled); }
TEST_F(GpuTest, TiledFloatInvariants) { gpu_invariants<float>(GpuKernel::Tiled); }
TEST_F(GpuTest, TiledDoubleInvariants) { gpu_invariants<double>(GpuKernel::Tiled); }
