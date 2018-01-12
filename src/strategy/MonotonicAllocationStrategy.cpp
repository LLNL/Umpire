#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

#include "umpire/util/AllocatorTraits.hpp"
#include "umpire/ResourceManager.hpp"

namespace umpire {

namespace strategy {

MonotonicAllocationStrategy::MonotonicAllocationStrategy(
    const std::string& name,
    int id,
    util::AllocatorTraits traits,
    std::vector<std::shared_ptr<AllocationStrategy> > providers) :
  AllocationStrategy(name, id),
  m_size(0)
{
  m_capacity = std::max(traits.m_maximum_size, traits.m_initial_size);
  m_allocator = providers[0];
  m_block = m_allocator->allocate(m_capacity);
}

void* 
MonotonicAllocationStrategy::allocate(size_t bytes)
{
  void* ret = static_cast<char*>(m_block) + bytes;
  m_size += bytes;

  if (m_size > m_capacity) {
    UMPIRE_ERROR("MonoticAllocationStrategy capacity exceeded " << m_size << " > " << m_capacity);
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  ResourceManager::getInstance().registerAllocation(ret, new util::AllocationRecord{ret, bytes, this->shared_from_this()});

  return ret;
}

void 
MonotonicAllocationStrategy::deallocate(void* ptr)
{
  UMPIRE_LOG(Info, "() doesn't do anything");
  // no op

  ResourceManager::getInstance().deregisterAllocation(ptr);
}

long 
MonotonicAllocationStrategy::getCurrentSize()
{
  UMPIRE_LOG(Debug, "() returning " << m_size);
  return m_size;
}

long 
MonotonicAllocationStrategy::getHighWatermark()
{
  UMPIRE_LOG(Debug, "() returning " << m_capacity);
  return m_capacity;
}

Platform 
MonotonicAllocationStrategy::getPlatform()
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
