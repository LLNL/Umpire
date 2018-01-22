#ifndef UMPIRE_MallocAllocator_HPP
#define UMPIRE_MallocAllocator_HPP

#include <cstdlib>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace alloc {

struct MallocAllocator
{
  void* allocate(size_t bytes) 
  {
    void* ret = ::malloc(bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

    if  (ret == nullptr) {
      UMPIRE_ERROR("malloc( bytes = " << bytes << " ) failed");
    } else {
      return ret;
    }
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    ::free(ptr);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_MallocAllocator_HPP
