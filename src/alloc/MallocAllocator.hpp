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

  void* malloc(size_t bytes);
  void* calloc(size_t bytes);
  void* realloc(void* ptr, size_t new_size);
  void free(void* ptr);
};

} // end of namespace alloc
} // end of namespace umpire

#include "umpire/alloc/MallocAllocator.inl"

#endif // UMPIRE_MallocAllocator_HPP
