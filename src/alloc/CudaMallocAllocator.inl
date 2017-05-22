#ifndef UMPIRE_CudaMallocAllocator_INL
#define UMPIRE_CudaMallocAllocator_INL

#include "umpire/alloc/CudaMallocAllocator.hpp"

#include <cuda.h>

namespace umpire {

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

} // end of namespace umpire

#endif
