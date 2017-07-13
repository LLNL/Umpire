#include "umpire/space/MemorySpace.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {

namespace space {

MemorySpace::MemorySpace(
    const std::string& name,
    alloc::MemoryAllocator* allocator) :
  m_descriptor(name),
  m_allocations(),
  m_default_allocator(allocator)
{
}

void* MemorySpace::allocate(size_t bytes)
{
  void* ptr = m_default_allocator->malloc(bytes);
  m_allocations[ptr] = m_default_allocator;
  ResourceManager::getInstance().registerAllocation(ptr, shared_from_this());

  return ptr;
}

void MemorySpace::free(void* ptr)
{
  m_default_allocator->free(ptr);
  ResourceManager::getInstance().deregisterAllocation(ptr);
}

void MemorySpace::getTotalSize()
{
}

void MemorySpace::getProperties(){}

void MemorySpace::getRemainingSize(){}

std::string MemorySpace::getDescriptor(){}

void MemorySpace::setDefaultAllocator(alloc::MemoryAllocator* allocator){}

alloc::MemoryAllocator& MemorySpace::getDefaultAllocator(){}

} // end of namespace space
} // end of namespace umpire
