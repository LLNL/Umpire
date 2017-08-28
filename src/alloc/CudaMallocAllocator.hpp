#ifndef UMPIRE_CudaMallocAllocator_HPP
#define UMPIRE_CudaMallocAllocator_HPP

#include <cuda_runtime_api.h>

namespace umpire {
namespace alloc {

struct CudaMallocAllocator
{
  void* allocate(size_t bytes)
  {
    void* ptr;
    ::cudaMalloc(&ptr, bytes);
    return ptr;
  }

  void deallocate(void* ptr)
  {
    ::cudaFree(ptr);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocAllocator_HPP
