#ifndef UMPIRE_MallocAllocator_HPP
#define UMPIRE_MallocAllocator_HPP

#include "umpire/MemoryAllocator.hpp"
#include "umpire/MemorySpace.hpp"

namespace umpire {

class MallocAllocator :
  public MemoryAllocator
{
 public:
  MallocAllocator();

  void* allocate(size_t bytes);

  void free(void* ptr);
};

} // end of namespace umpire

#include "umpire/MallocAllocator.inl"

#endif // UMPIRE_MallocAllocator_HPP

