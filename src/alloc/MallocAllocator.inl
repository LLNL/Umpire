#ifndef UMPIRE_MallocAllocator_INL
#define UMPIRE_MallocAllocator_INL

#include "umpire/MallocAllocator.hpp"

#include <cstdlib>

namespace umpire {

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

} // end of namespace umpire

#endif
