//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/Synchronizer.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {
namespace strategy {

Synchronizer::Synchronizer(const std::string& name, int id, Allocator allocator, camp::resources::Resource r, bool sync_before_alloc, bool sync_before_dealloc)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "Synchronizer"},
      m_allocator{allocator.getAllocationStrategy()},
      m_resource{r},
      m_sync_before_alloc{sync_before_alloc},
      m_sync_before_dealloc{sync_before_dealloc}
{
}

void* Synchronizer::allocate(std::size_t bytes)
{
  if (m_sync_before_alloc) {
    m_resource.wait();
  }
  return m_allocator->allocate_internal(bytes);
}

void Synchronizer::deallocate(void* p, std::size_t size)
{
  if (m_sync_before_dealloc) {
    m_resource.wait();
  }
  m_allocator->deallocate_internal(p, size);
}

Platform Synchronizer::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits Synchronizer::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // namespace strategy
} // namespace umpire
