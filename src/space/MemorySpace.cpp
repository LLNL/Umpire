#include "umpire/space/MemorySpace.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {

namespace space {

template<typename _allocator>
MemorySpace<_allocator>::MemorySpace(
    const std::string& name) :
  m_descriptor(name),
  m_allocator()
{
}

template<typename _allocator>
void* MemorySpace<_allocator>::allocate(size_t bytes)
{
  void* ptr = m_allocator.allocate(bytes);
  //m_allocations[ptr] = m_allocator;
  ResourceManager::getInstance().registerAllocation(ptr, shared_from_this());

  return ptr;
}

template<typename _allocator>
void MemorySpace<_allocator>::free(void* ptr)
{
  m_allocator.deallocate(ptr);
  ResourceManager::getInstance().deregisterAllocation(ptr);
}

template<typename _allocator>
void MemorySpace<_allocator>::getTotalSize()
{
}

template<typename _allocator>
void MemorySpace<_allocator>::getProperties(){}

template<typename _allocator>
void MemorySpace<_allocator>::getRemainingSize(){}

template<typename _allocator>
std::string MemorySpace<_allocator>::getDescriptor(){}

template<typename _allocator>
void MemorySpace<_allocator>::setDefaultAllocator(alloc::MemoryAllocator* allocator){}

} // end of namespace space
} // end of namespace umpire
