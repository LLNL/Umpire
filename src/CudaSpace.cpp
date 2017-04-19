#include "umpire/GpuSpace.hpp"

#include "CudaMallocAllocator.hpp"

namespace umpire {

GpuSpace::GpuSpace()
{
  m_default_allocator = new CudaMallocAllocator();
}

void* GpuSpace::allocate(size_t bytes)
{
  return m_default_allocator->allocate(bytes);
}

void GpuSpace::free(void* ptr)
{
  m_default_allocator->free(ptr);
}

void GpuSpace::getTotalSize()
{}

void GpuSpace::getProperties(){}

void GpuSpace::getRemainingSize(){}

std::string GpuSpace::getDescriptor(){}

void GpuSpace::setDefaultAllocator(MemoryAllocator allocator){}

MemoryAllocator GpuSpace::getDefaultAllocator(){}

std::vector<MemoryAllocator> GpuSpace::getAllocators(){}

} // end of namespace umpire
