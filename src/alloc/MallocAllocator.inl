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
MallocAllocator::malloc(size_t bytes)
{
  return ::malloc(bytes);
}

inline
void*
MallocAllocator::calloc(size_t bytes)
{
  return ::calloc(bytes, 1);
}

inline
void*
MallocAllocator::realloc(void* ptr, size_t new_size)
{
  return ::realloc(ptr, new_size);
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
