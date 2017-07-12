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
PosixMemalignAllocator::allocate(size_t bytes)
{
  void* ptr = nullptr;
  ::posix_memalign(&ptr, 64, bytes);
  return ptr;
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
