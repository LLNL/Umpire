#ifndef UMPIRE_MallocAllocator_INL
#define UMPIRE_MallocAllocator_INL

#include "umpire/alloc/MallocAllocator.hpp"

#include <cstdlib>

namespace umpire {
namespace alloc {

inline
MallocAllocator::MallocAllocator()
{
}


inline
void*
MallocAllocator::allocate(size_t bytes)
{
  return ::malloc(bytes);
}

inline
void
MallocAllocator::free(void* ptr)
{
  ::free(ptr);
}

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_MallocAllocator_INL
