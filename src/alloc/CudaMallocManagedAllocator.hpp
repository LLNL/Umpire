#ifndef UMPIRE_CudaMallocManagedAllocator_HPP
#define UMPIRE_CudaMallocManagedAllocator_HPP

#include <cuda_runtime_api.h>

namespace umpire {
namespace alloc {

struct CudaMallocManagedAllocator
{
  void* allocate(size_t bytes)
  {
    void* ptr = nullptr;
    cudaError_t error = ::cudaMallocManaged(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaMallocManaged( bytes = " << bytes << " ) failed with error: " << cudaGetErrorString(error));
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

#endif // UMPIRE_CudaMallocManagedAllocator_HPP
