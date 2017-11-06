#include "umpire/strategy/MonotonicAllocationStrategy.hpp"

#include "umpire/util/Macros.hpp"

#include "umpire/AllocatorTraits.hpp"

namespace umpire {

namespace strategy {

MonotonicAllocationStrategy::MonotonicAllocationStrategy(
    AllocatorTraits traits,
    std::vector<std::shared_ptr<AllocationStrategy> > providers) :
  m_size(0),
  m_sizes()
{
  m_capacity = std::max(traits.m_maximum_size, traits.m_initial_size);
  m_allocator = providers[0];
  m_block = m_allocator->allocate(m_capacity);
}

void* 
MonotonicAllocationStrategy::allocate(size_t bytes)
{
  void* ret = m_block + bytes;
  m_size += bytes;
  m_sizes[ret] = bytes;

  if (m_size > m_capacity) {
    UMPIRE_ERROR("MonoticAllocationStrategy capacity exceeded " << m_size << " > " << m_capacity);
  }

  return ret;
}

void 
MonotonicAllocationStrategy::deallocate(void* ptr)
{
  UMPIRE_LOG("MonoticAllocationStrategy: deallocate doesn't do anything");
  // no op
}

size_t 
MonotonicAllocationStrategy::getSize(void* ptr)
{
  auto it = m_sizes.find(ptr);
  
  if (it != m_sizes.end()) {
    return it->second;
  } else {
    return m_allocator->getSize(ptr);
  }
}

long 
MonotonicAllocationStrategy::getCurrentSize()
{
  return m_size;
}

long 
MonotonicAllocationStrategy::getHighWatermark()
{
  return m_capacity;
}

Platform 
MonotonicAllocationStrategy::getPlatform()
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
