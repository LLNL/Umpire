#ifndef UMPIRE_MallocAllocator_HPP
#define UMPIRE_MallocAllocator_HPP

#include "umpire/alloc/MemoryAllocator.hpp"

namespace umpire {
namespace alloc {

class MallocAllocator :
  public MemoryAllocator
{
 public:
  MallocAllocator();

  void* allocate(size_t bytes);

  void free(void* ptr);
};

} // end of namespace alloc
} // end of namespace umpire

#include "umpire/alloc/MallocAllocator.inl"

#endif // UMPIRE_MallocAllocator_HPP
