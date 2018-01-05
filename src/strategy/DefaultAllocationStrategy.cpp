#include "umpire/strategy/DefaultAllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

DefaultAllocationStrategy::DefaultAllocationStrategy(std::shared_ptr<AllocationStrategy> allocator) :
  m_allocator(allocator)
{
}

void* 
DefaultAllocationStrategy::allocate(size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");
  return m_allocator->allocate(bytes);
}

void 
DefaultAllocationStrategy::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  return m_allocator->deallocate(ptr);
}

long 
DefaultAllocationStrategy::getCurrentSize()
{
  return m_allocator->getCurrentSize();
}

long 
DefaultAllocationStrategy::getHighWatermark()
{
  return m_allocator->getHighWatermark();
}

Platform 
DefaultAllocationStrategy::getPlatform()
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
