#pragma once
#include "cuda_check.hpp"
namespace orbitlab {
struct CudaEvent {
 cudaEvent_t value=nullptr;
 CudaEvent() { CUDA_CHECK(cudaEventCreate(&value)); }
 ~CudaEvent() { if(value) CUDA_CLEANUP(cudaEventDestroy(value)); }
 CudaEvent(const CudaEvent&)=delete;CudaEvent& operator=(const CudaEvent&)=delete;
};
}
