#ifndef UMPIRE_MemorySpace_INL
#define UMPIRE_MemorySpace_INL

#include "umpire/space/MemorySpace.hpp"
#include "umpire/ResourceManager.hpp"

#include <memory>

namespace umpire {
namespace space {

template<typename _allocator>
MemorySpace<_allocator>::MemorySpace() :
  m_allocator(),
  m_allocations(),
  m_current_size(0l),
  m_highwatermark(0l)
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
  if (allocation != m_allocations.end())
    m_current_size -= allocation->second.m_size;
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


} // end of namespace space
} // end of namespace umpire
#endif // UMPIRE_MemorySpace_INL
