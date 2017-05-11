#ifndef UMPIRE_Pool_HPP
#define UMPIRE_Pool_HPP

#include "umpire/Allocator.hpp"

#include "umpire/space/MemorySpace.hpp"
#include "umpire/alloc/MemoryAllocator.hpp"

namespace umpire {
namespace alloc {

template <typename T>
class Pool : public Allocator {
  public:
    Pool();
    Pool(space::MemorySpace* space);

    void* allocate(size_t bytes);
    void free(void* ptr);

  private:
    void init();

    MemoryAllocator m_allocator;
    space::MemorySpace* m_space;

    T* m_pointers[32];
    int m_lengths[32];

};

}
}

#endif // UMPIRE_Pool_HPP
