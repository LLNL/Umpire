#ifndef UMPIRE_MallocAllocator_HPP
#define UMPIRE_MallocAllocator_HPP

#include <cstdlib>

namespace umpire {
namespace alloc {

struct MallocAllocator
{
  void* allocate(size_t bytes) 
  {
    void* ret = ::malloc(bytes);
    if  (ret == nullptr) {
      UMPIRE_ERROR("malloc returned NULL, allocating " << bytes << " bytes.");
    } else {
      return ret;
    }
  }

  void deallocate(void* ptr)
  {
    ::free(ptr);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_MallocAllocator_HPP
