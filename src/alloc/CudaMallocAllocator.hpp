#ifndef UMPIRE_CudaMallocAllocator_HPP
#define UMPIRE_CudaMallocAllocator_HPP

#include "umpire/alloc/MemoryAllocator.hpp"

namespace umpire {
namespace alloc {

class CudaMallocAllocator :
  public alloc::MemoryAllocator
{
  public:
  CudaMallocAllocator();

  void* allocate(size_t bytes);

  void free(void* ptr);
};

} // end of namespace alloc
} // end of namespace umpire

#include "umpire/alloc/CudaMallocAllocator.inl"

#endif // UMPIRE_CudaMallocAllocator_HPP
