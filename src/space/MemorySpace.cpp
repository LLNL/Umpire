#include "umpire/space/MemorySpace.hpp"

namespace umpire {

namespace space {

MemorySpace::MemorySpace()
{
}

MemorySpace::MemorySpace(alloc::MemoryAllocator* allocator) :
  m_default_allocator(allocator)
{
}

void* MemorySpace::allocate(size_t bytes)
{
  return m_default_allocator->allocate(bytes);
}

void MemorySpace::free(void* ptr)
{
  m_default_allocator->free(ptr);
}

void MemorySpace::getTotalSize()
{}

void MemorySpace::getProperties(){}

void MemorySpace::getRemainingSize(){}

std::string MemorySpace::getDescriptor(){}

void MemorySpace::setDefaultAllocator(alloc::MemoryAllocator* allocator){}

alloc::MemoryAllocator& MemorySpace::getDefaultAllocator(){}

std::vector<alloc::MemoryAllocator*> MemorySpace::getAllocators(){}

} // end of namespace space
} // end of namespace umpire
