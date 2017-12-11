#include "umpire/strategy/SimpoolAllocationStrategy.hpp"

namespace umpire {
namespace strategy {

SimpoolAllocationStrategy::SimpoolAllocationStrategy(
    util::AllocatorTraits,
    std::vector<std::shared_ptr<AllocationStrategy> > providers) :
  m_current_size(0),
  m_highwatermark(0)
{
  m_allocator = providers[0];
  dpa = new DynamicPoolAllocator<>(m_allocator);
}

void*
SimpoolAllocationStrategy::allocate(size_t bytes) { 
  return dpa->allocate(bytes);
}

void 
SimpoolAllocationStrategy::deallocate(void* ptr) {
  dpa->deallocate(ptr);
}

long 
SimpoolAllocationStrategy::getCurrentSize()
{ 
  return m_allocator->getCurrentSize(); 
}

long 
SimpoolAllocationStrategy::getHighWatermark()
{ 
  return m_allocator->getHighWatermark();
}

size_t 
SimpoolAllocationStrategy::getSize(void* ptr)
{ 
  return m_allocator->getSize(ptr);
}

Platform 
SimpoolAllocationStrategy::getPlatform()
{ 
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
