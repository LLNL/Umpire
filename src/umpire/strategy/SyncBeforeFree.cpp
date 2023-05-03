//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/SyncBeforeFree.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {
namespace strategy {

SyncBeforeFree::SyncBeforeFree(const std::string& name, int id, Allocator allocator, camp::resources::Resource r)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "SyncBeforeFree"},
      m_allocator{allocator.getAllocationStrategy()},
      m_resource{r}
{
}

void* SyncBeforeFree::allocate(std::size_t bytes)
{
  return m_allocator->allocate_internal(bytes);
}

void SyncBeforeFree::deallocate(void* p, std::size_t size)
{
  m_resource.wait();
  m_allocator->deallocate_internal(p, size);
}

Platform SyncBeforeFree::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits SyncBeforeFree::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // namespace strategy
} // namespace umpire