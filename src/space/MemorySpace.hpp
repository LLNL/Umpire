#ifndef UMPIRE_MemorySpace_HPP
#define UMPIRE_MemorySpace_HPP

#include "umpire/AllocatorInterface.hpp"

#include "umpire/util/AllocationRecord.hpp"

#include <memory>
#include <unordered_map>

namespace umpire {
namespace space {

template <typename _allocator>
class MemorySpace : 
  public AllocatorInterface
{

  public: 
    MemorySpace();

    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    size_t size(void* ptr);

    long getCurrentSize();
    long getHighWatermark();

  protected: 
    _allocator m_allocator;

    std::unordered_map<void*, util::AllocationRecord> m_allocations;

    long m_current_size;
    long m_highwatermark;
};

} // end of namespace space
} // end of namespace umpire

#include "umpire/space/MemorySpace.inl"

#endif // UMPIRE_MemorySpace_HPP
