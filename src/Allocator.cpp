#include "umpire/Allocator.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

Allocator::Allocator(std::shared_ptr<strategy::AllocationStrategy>& allocator):
  m_allocator(allocator)
{
}

void*
Allocator::allocate(size_t bytes)
{
  UMPIRE_LOG(Debug, "(" << bytes << ")");
  return m_allocator->allocate(bytes);
}

void
Allocator::deallocate(void* ptr)
{
  UMPIRE_ASSERT("Deallocate called with nullptr" && ptr);
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  m_allocator->deallocate(ptr);
}

size_t
Allocator::getSize(void* ptr)
{
  UMPIRE_LOG(Debug, "(" << ptr << ")");
  return ResourceManager::getInstance().getSize(ptr);
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

std::string
Allocator::getName()
{
  return m_allocator->getName();
}

int
Allocator::getId()
{
  return m_allocator->getId();
}

std::shared_ptr<strategy::AllocationStrategy>
Allocator::getAllocationStrategy()
{
  UMPIRE_LOG(Debug, "() returning " << m_allocator);
  return m_allocator;
}

} // end of namespace umpire
