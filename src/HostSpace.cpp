#include "umpire/HostSpace.hpp"

#include "MallocAllocator.hpp"

namespace umpire {

HostSpace::HostSpace()
{
  m_default_allocator = new MallocAllocator();
}

void* HostSpace::allocate(size_t bytes)
{
  return m_default_allocator->allocate(bytes);
}

void HostSpace::free(void* ptr)
{
  m_default_allocator->free(ptr);
}

void HostSpace::getTotalSize()
{}

  void HostSpace::getProperties(){}

  void HostSpace::getRemainingSize(){}

  std::string HostSpace::getDescriptor(){}

  void HostSpace::setDefaultAllocator(MemoryAllocator& allocator){}

  MemoryAllocator& HostSpace::getDefaultAllocator(){}

  std::vector<MemoryAllocator*> HostSpace::getAllocators(){}

} // end of namespace umpire
