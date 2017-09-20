#include "umpire/Allocator.hpp"

namespace umpire {

Allocator::Allocator(std::shared_ptr<umpire::AllocatorInterface>& allocator):
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
Allocator::size(void* ptr)
{
  return m_allocator->size(ptr);
}

} // end of namespace umpire
