#ifndef UMPIRE_DefaultMemoryResource_HPP
#define UMPIRE_DefaultMemoryResource_HPP

#include "umpire/resource/MemoryResource.hpp"

#include <memory>
#include <unordered_map>

#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

template <typename _allocator>
class DefaultMemoryResource :
  public MemoryResource
{
  public: 
    DefaultMemoryResource(Platform platform = Platform::cpu);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    size_t getSize(void* ptr);
    long getCurrentSize();
    long getHighWatermark();

    Platform getPlatform();

  protected: 
    _allocator m_allocator;

    std::unordered_map<void*, util::AllocationRecord> m_allocations;

    long m_current_size;
    long m_highwatermark;

    Platform m_platform;
};

} // end of namespace resource
} // end of namespace umpire

#include "umpire/resource/DefaultMemoryResource.inl"

#endif // UMPIRE_DefaultMemoryResource_HPP
