#ifndef UMPIRE_CudaMallocAllocator_HPP
#define UMPIRE_CudaMallocAllocator_HPP

#include "umpire/MemoryAllocator.hpp"
#include "umpire/MemorySpace.hpp"

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

#include "umpire/CudaMallocAllocator.inl"

#endif // UMPIRE_CudaMallocAllocator_HPP

