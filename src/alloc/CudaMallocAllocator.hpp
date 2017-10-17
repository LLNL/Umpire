#ifndef UMPIRE_CudaMallocAllocator_HPP
#define UMPIRE_CudaMallocAllocator_HPP

#include <cuda_runtime_api.h>

namespace umpire {
namespace alloc {

struct CudaMallocAllocator
{
  void* allocate(size_t bytes)
  {
    void* ptr = nullptr;
    cudaError_t error = ::cudaMalloc(&ptr, bytes);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaMalloc failed allocating " << bytes << "bytes, with: " << cudaGetErrorString(error));
    } else {
      return ptr;
    }
  }

  void deallocate(void* ptr)
  {
    ::cudaFree(ptr);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocAllocator_HPP
