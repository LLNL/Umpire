#ifndef UMPIRE_AllocationRecord_HPP
#define UMPIRE_AllocationRecord_HPP

#include "umpire/alloc/MemoryAllocator.hpp"

namespace umpire {

struct AllocationRecord
{
  void* m_ptr;
  size_t m_size;

  alloc::MemoryAllocator* m_allocator;
};

} // end of namespace umpire

#endif // UMPIRE_AllocationRecord_HPP
