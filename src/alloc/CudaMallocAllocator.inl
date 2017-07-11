#ifndef UMPIRE_CudaMallocAllocator_INL
#define UMPIRE_CudaMallocAllocator_INL

#include "umpire/alloc/CudaMallocAllocator.hpp"

#include <cuda_runtime.h>

namespace umpire {
namespace alloc {

inline
CudaMallocAllocator::CudaMallocAllocator()
{
}


inline
void*
CudaMallocAllocator::allocate(size_t bytes)
{
  void* ptr;
  cudaMalloc(&ptr, bytes);
  return ptr;
}


inline
void
CudaMallocAllocator::free(void* ptr)
{
  cudaFree(ptr);
}

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocAllocator_INL
