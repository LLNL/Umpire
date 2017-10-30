#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {
  
AllocationStrategy::AllocationStrategy(std::shared_ptr<umpire::AllocatorInterface>& alloc):
  m_allocator(alloc)
{
}

Platform
AllocationStrategy::getPlatform()
{
  return m_allocator->getPlatform();
}

size_t 
AllocationStrategy::getSize(void* ptr)
{
  return m_allocator->getSize(ptr);
}

} // end of namespace strategy
} // end of namespace umpire
