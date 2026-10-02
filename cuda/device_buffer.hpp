#pragma once
#include "cuda_check.hpp"
namespace orbitlab {
template<class R> struct DeviceBuffer {
 R* data=nullptr;
 explicit DeviceBuffer(std::size_t count) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(R))); }
 ~DeviceBuffer() { if(data) CUDA_CLEANUP(cudaFree(data)); }
 DeviceBuffer(const DeviceBuffer&)=delete;
 DeviceBuffer& operator=(const DeviceBuffer&)=delete;
};
}
