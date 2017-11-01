#include "umpire/Allocator.hpp"

namespace umpire {

Allocator::Allocator(std::shared_ptr<strategy::AllocationStrategy>& allocator):
  m_allocator(allocator)
{
}

void*
Allocator::allocate(size_t bytes)
{
  return m_allocator->allocate(bytes);
}

void
Allocator::deallocate(void* ptr)
{
  m_allocator->deallocate(ptr);
}

size_t
Allocator::getSize(void* ptr)
{
  return m_allocator->getSize(ptr);
}

size_t
Allocator::getHighWatermark()
{
  return m_allocator->getHighWatermark();
}

size_t
Allocator::getCurrentSize()
{
  return m_allocator->getCurrentSize();
}

std::shared_ptr<strategy::AllocationStrategy>
Allocator::getAllocationStrategy()
{
  return m_allocator;
}

} // end of namespace umpire
