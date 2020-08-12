//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/AllocationPrefetcher.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/op/MemoryOperationRegistry.hpp"

namespace umpire {
namespace strategy {

AllocationPrefetcher::AllocationPrefetcher(const std::string& name, int id,
                                           Allocator allocator, int device_id)
    : AllocationStrategy(name, id),
      m_allocator{allocator.getAllocationStrategy()},
      m_device{device_id}
{
  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  m_prefetch_operation = op_registry.find("PREFETCH", m_allocator, m_allocator);
}

void* AllocationPrefetcher::allocate(std::size_t bytes)
{
  void* ptr = m_allocator->allocate(bytes);

  m_prefetch_operation->apply(ptr, nullptr, m_device, bytes);

  return ptr;
}

void AllocationPrefetcher::deallocate(void* ptr)
{
  m_allocator->deallocate(ptr);
}

Platform AllocationPrefetcher::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits AllocationPrefetcher::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire
