#include "umpire/strategy/SimpoolAllocationStrategy.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

SimpoolAllocationStrategy::SimpoolAllocationStrategy(
    util::AllocatorTraits,
    std::vector<std::shared_ptr<AllocationStrategy> > providers) :
  dpa(nullptr),
  m_current_size(0),
  m_highwatermark(0),
  m_allocator(),
  m_allocations()
{
  m_allocator = providers[0];
  dpa = new DynamicPoolAllocator<>(m_allocator);
}

void*
SimpoolAllocationStrategy::allocate(size_t bytes) { 
  void* ptr = dpa->allocate(bytes);
  ResourceManager::getInstance().registerAllocation(ptr, new util::AllocationRecord{ptr, bytes, this->shared_from_this()});

  m_allocations[ptr] = {ptr, bytes};

  m_current_size += bytes;
  if (m_current_size > m_highwatermark)
    m_highwatermark = m_current_size;

  return ptr;
}

void 
SimpoolAllocationStrategy::deallocate(void* ptr) {
  dpa->deallocate(ptr);
  ResourceManager::getInstance().deregisterAllocation(ptr);

  auto allocation = m_allocations.find(ptr);
  if (allocation != m_allocations.end()) {
    m_current_size -= allocation->second.m_size;
    m_allocations.erase(allocation);
  }
}

long 
SimpoolAllocationStrategy::getCurrentSize()
{ 
  return dpa->totalSize(); 
}

long 
SimpoolAllocationStrategy::getHighWatermark()
{ 
  return m_highwatermark;
}

size_t 
SimpoolAllocationStrategy::getSize(void* ptr)
{ 
  auto allocation = m_allocations.find(ptr);
  if (allocation == m_allocations.end()) {
    UMPIRE_ERROR("size for " << ptr << " not found");
  }

  return allocation->second.m_size;
}

Platform 
SimpoolAllocationStrategy::getPlatform()
{ 
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
