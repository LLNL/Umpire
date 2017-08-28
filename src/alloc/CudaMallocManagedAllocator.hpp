#ifndef UMPIRE_CudaMallocManagedAllocator_HPP
#define UMPIRE_CudaMallocManagedAllocator_HPP

#include <cuda_runtime_api.h>

namespace umpire {
namespace alloc {

struct CudaMallocManagedAllocator :
{
  void* allocate(size_t bytes)
  {
    void* ptr;
    ::cudaMallocManaged(&ptr, bytes);
    return ptr;
  }

  void deallocate(void* ptr)
  {
    ::cudaFree(ptr);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocManagedAllocator_HPP
