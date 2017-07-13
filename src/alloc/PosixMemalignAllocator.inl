#ifndef UMPIRE_PosixMemalignAllocator_INL
#define UMPIRE_PosixMemalignAllocator_INL

#include "umpire/alloc/PosixMemalignAllocator.hpp"

#include <cstdlib>

namespace umpire {
namespace alloc {

inline 
PosixMemalignAllocator::PosixMemalignAllocator()
{
}

inline 
void*
PosixMemalignAllocator::malloc(size_t bytes)
{
  void* ptr = nullptr;
  ::posix_memalign(&ptr, 64, bytes);
  return ptr;
}

inline 
void*
PosixMemalignAllocator::calloc(size_t bytes)
{
  void* ptr = nullptr;
  ::posix_memalign(&ptr, 64, bytes);
  ::memset(ptr, 0, bytes);
  return ptr;
}

inline 
void*
PosixMemalignAllocator::realloc(void* ptr, size_t new_size)
{
  /* TODO: check alignment of this new allocation */
  void* new_ptr = ::realloc(ptr, new_size);
  return new_ptr;
}

inline
void
PosixMemalignAllocator::free(void* ptr)
{
  ::free(ptr);
}



} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_PosixMemalignAllocator_INL
