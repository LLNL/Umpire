#ifndef UMPIRE_MallocAllocator_HPP
#define UMPIRE_MallocAllocator_HPP

#include <cstdlib>

namespace umpire {
namespace alloc {

struct MallocAllocator :
{
  void* allocate(size_t bytes) 
  {
    return std::malloc(bytes);
  }

  void* deallocate(void* ptr)
  {
    std::free(ptr);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_MallocAllocator_HPP
