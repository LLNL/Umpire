#ifndef UMPIRE_Pool_HPP
#define UMPIRE_Pool_HPP

#include "umpire/Allocator.hpp"

#include "umpire/space/MemorySpace.hpp"
#include "umpire/alloc/MallocAllocator.hpp"

#include <memory>

namespace umpire {
namespace alloc {

template <typename allocator=MallocAllocator>
class Pool : public Allocator {
  public:
    Pool();
    Pool(std::shared_ptr<space::MemorySpace> space);

    void* allocate(size_t bytes);
    void free(void* ptr);

  private:
    void init();

    allocator m_allocator;
    std::shared_ptr<space::MemorySpace> m_space;

    void* m_pointers[32];
    int m_lengths[32];

};

}
}

#include "umpire/alloc/Pool.inl"

#endif // UMPIRE_Pool_HPP
