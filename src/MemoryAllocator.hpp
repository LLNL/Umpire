#ifndef UMPIRE_MemoryAllocator_HPP
#define UMPIRE_MemoryAllocator_HPP

#include <cstddef>

namespace umpire {

class MemoryAllocator {
  public:
  virtual void* allocate(size_t bytes) = 0;

  virtual void free(void* ptr) = 0;
};

} // end of namespace umpire

#endif // UMPIRE_MemoryAllocator_HPP
