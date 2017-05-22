#ifndef UMPIRE_CudaMallocAllocator_HPP
#define UMPIRE_CudaMallocAllocator_HPP

#include "umpire/alloc/MemoryAllocator.hpp"

namespace umpire {

class CudaMallocAllocator :
  public MemoryAllocator
{
 public:
  CudaMallocAllocator();

  void* alloc(size_t bytes);

  void free(void* ptr);
};

} // end of namespace umpire

#include "umpire/alloc/CudaMallocAllocator.inl"

#endif // UMPIRE_CudaMallocAllocator_HPP

