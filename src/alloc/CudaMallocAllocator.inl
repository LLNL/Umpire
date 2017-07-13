#ifndef UMPIRE_CudaMallocAllocator_INL
#define UMPIRE_CudaMallocAllocator_INL

#include "umpire/alloc/CudaMallocAllocator.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace alloc {

inline
CudaMallocAllocator::CudaMallocAllocator()
{
}


inline
void*
CudaMallocAllocator::malloc(size_t bytes)
{
  void* ptr;
  ::cudaMalloc(&ptr, bytes);
  return ptr;
}

inline
void*
CudaMallocAllocator::calloc(size_t bytes)
{
  void* ptr;
  ::cudaMalloc(&ptr, bytes);
  ::cudaMemset(ptr, 0, bytes);
  return ptr;
}

inline
void*
CudaMallocAllocator::realloc(void* ptr, size_t new_size)
{
  void* new_ptr;
  ::cudaMalloc(&ptr, new_size);
  /* TODO: this should be the actual size of the old allocation */
  ::cudaMemcpy(new_ptr, ptr, new_size, cudaMemcpyDeviceToDevice);
  ::cudaFree(ptr);
  return new_ptr;
}

inline
void
CudaMallocAllocator::free(void* ptr)
{
  ::cudaFree(ptr);
}

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocAllocator_INL
