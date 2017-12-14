#ifndef UMPIRE_DefaultMemoryResource_INL
#define UMPIRE_DefaultMemoryResource_INL

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#include <memory>
#include <sstream>

namespace umpire {
namespace resource {

template<typename _allocator>
DefaultMemoryResource<_allocator>::DefaultMemoryResource(Platform platform) :
  m_allocator(),
  m_allocations(),
  m_current_size(0l),
  m_highwatermark(0l),
  m_platform(platform)
{
}

template<typename _allocator>
void* DefaultMemoryResource<_allocator>::allocate(size_t bytes)
{
  void* ptr = m_allocator.allocate(bytes);
  ResourceManager::getInstance().registerAllocation(ptr, new util::AllocationRecord{ptr, bytes, this->shared_from_this()});

  m_allocations[ptr] = {ptr, bytes};

  m_current_size += bytes;
  if (m_current_size > m_highwatermark)
    m_highwatermark = m_current_size;

  return ptr;
}

template<typename _allocator>
void DefaultMemoryResource<_allocator>::deallocate(void* ptr)
{
  m_allocator.deallocate(ptr);
  ResourceManager::getInstance().deregisterAllocation(ptr);

  auto allocation = m_allocations.find(ptr);
  if (allocation != m_allocations.end()) {
    m_current_size -= allocation->second.m_size;
    m_allocations.erase(allocation);
  }
}

template<typename _allocator>
size_t DefaultMemoryResource<_allocator>::getSize(void* ptr)
{
  auto allocation = m_allocations.find(ptr);
  if (allocation == m_allocations.end()) {
    std::stringstream e;
    e << "size for " << ptr << " not found" << std::endl;
    UMPIRE_ERROR(e.str());
    return(0);
  }

  return allocation->second.m_size;
}

template<typename _allocator>
long DefaultMemoryResource<_allocator>::getCurrentSize()
{
  return m_current_size;
}

template<typename _allocator>
long DefaultMemoryResource<_allocator>::getHighWatermark()
{
  return m_highwatermark;
}

template<typename _allocator>
Platform DefaultMemoryResource<_allocator>::getPlatform()
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_DefaultMemoryResource_INL
