#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdlib>
namespace orbitlab {
inline void check_cuda(cudaError_t status,const char* call,const char* file,int line) {
 if(status!=cudaSuccess) throw std::runtime_error(std::string(file)+":"+std::to_string(line)+" "+call+": "+cudaGetErrorString(status));
}
inline void cleanup_cuda(cudaError_t status,const char* call,const char* file,int line) noexcept {
 try { check_cuda(status,call,file,line); }
 catch(const std::exception& e) { std::cerr<<"CUDA cleanup failed: "<<e.what()<<"\n";std::abort(); }
}
}
#define CUDA_CHECK(call) ::orbitlab::check_cuda((call),#call,__FILE__,__LINE__)
#define CUDA_CLEANUP(call) ::orbitlab::cleanup_cuda((call),#call,__FILE__,__LINE__)
