#ifndef UMPIRE_PosixMemalignAllocator_HPP
#define UMPIRE_PosixMemalignAllocator_HPP

#include "umpire/alloc/MemoryAllocator.hpp"

namespace umpire {
namespace alloc {

class PosixMemalignAllocator :
  public MallocAllocator
{
  public:
    PosixMemalignAllocator();

    void allocate(size_t bytes);

    void free(void * ptr);
}

} // end of namespace alloc
} // end of namespace umpire

#include "umpire/alloc/PosixMemalignAllocator.inl"

#endif // UMPIRE_PosixMemalignAllocator_HPP
