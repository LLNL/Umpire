//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/AllocationPrefetcher.hpp"
#include "umpire/op/MemoryOperationRegistry.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {
namespace strategy {

AllocationPrefetcher::AllocationPrefetcher(
    const std::string& name,
    int id,
    Allocator allocator,
    int device_id) :
  AllocationStrategy(name, id),
  m_allocator{allocator.getAllocationStrategy()},
  m_device{device_id}
{
  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  m_prefetch_operation = op_registry.find(
      "PREFETCH",
      m_allocator,
      m_allocator);
}

void* AllocationPrefetcher::allocate(std::size_t bytes)
{
  void* ptr = m_allocator->allocate(bytes);

  m_prefetch_operation->apply(
      ptr,
      nullptr,
      m_device,
      bytes);

  return ptr;
}

void AllocationPrefetcher::deallocate(void* ptr)
{
  m_allocator->deallocate(ptr);
}

std::size_t AllocationPrefetcher::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t AllocationPrefetcher::getHighWatermark() const noexcept
{
  return 0;
}

Platform AllocationPrefetcher::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits
AllocationPrefetcher::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

AllocationStrategy*
AllocationPrefetcher::getAllocationResource() noexcept
{
  return m_allocator->getAllocationResource();
}

} // end of namespace strategy
} // end of namespace umpire
