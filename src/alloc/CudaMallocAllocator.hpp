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
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaMalloc( bytes = " << bytes << " ) failed with error: " << cudaGetErrorString(error));
    } else {
      return ptr;
    }
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    cudaError_t error = ::cudaFree(ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaFree( ptr = " << ptr << " ) failed with error: " << cudaGetErrorString(error));
    }
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocAllocator_HPP
