#ifndef UMPIRE_PosixMemalignAllocator_HPP
#define UMPIRE_PosixMemalignAllocator_HPP

namespace umpire {
namespace alloc {

class PosixMemalignAllocator : public MemoryAllocator {
  public:
  virtual void* malloc(size_t bytes) = 0;
  virtual void* calloc(size_t bytes) = 0;
  virtual void* realloc(void* ptr, size_t new_size) = 0;
  virtual void free(void* ptr) = 0;
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_PosixMemalignAllocator_HPP
