#ifndef UMPIRE_MemorySpace_INL
#define UMPIRE_MemorySpace_INL

#include "umpire/space/MemorySpace.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#include <memory>
#include <sstream>

namespace umpire {
namespace space {

template<typename _allocator>
MemorySpace<_allocator>::MemorySpace(Platform platform = Platform::cpu) :
  m_allocator(),
  m_allocations(),
  m_current_size(0l),
  m_highwatermark(0l),
  m_platform(platform)
{
}

template<typename _allocator>
void* MemorySpace<_allocator>::allocate(size_t bytes)
{
  void* ptr = m_allocator.allocate(bytes);
  ResourceManager::getInstance().registerAllocation(ptr, this->shared_from_this());

  m_allocations[ptr] = {ptr, bytes};

  m_current_size += bytes;
  if (m_current_size > m_highwatermark)
    m_highwatermark = m_current_size;

  return ptr;
}

template<typename _allocator>
void MemorySpace<_allocator>::deallocate(void* ptr)
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
size_t MemorySpace<_allocator>::size(void* ptr)
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
long MemorySpace<_allocator>::getCurrentSize()
{
  return m_current_size;
}

template<typename _allocator>
long MemorySpace<_allocator>::getHighWatermark()
{
  return m_highwatermark;
}

template<typename _allocator>
Platform getPlatform()
{
  return m_platform;
}

} // end of namespace space
} // end of namespace umpire
#endif // UMPIRE_MemorySpace_INL
