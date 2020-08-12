//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/AllocationTracker.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

AllocationTracker::AllocationTracker(
    std::unique_ptr<AllocationStrategy>&& allocator) noexcept
    : AllocationStrategy(allocator->getName(), allocator->getId()),
      mixins::Inspector(),
      m_allocator(std::move(allocator))
{
}

void* AllocationTracker::allocate(std::size_t bytes)
{
  void* ptr = m_allocator->allocate(bytes);

  UMPIRE_LOG(Debug,
             "Tracking " << bytes << " bytes for " << m_allocator->getName());

  registerAllocation(ptr, bytes, this);

  return ptr;
}

void AllocationTracker::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug,
             "Untracking address " << ptr << " for " << m_allocator->getName());

  deregisterAllocation(ptr, this);
  m_allocator->deallocate(ptr);
}

void AllocationTracker::release()
{
  m_allocator->release();
}

std::size_t AllocationTracker::getCurrentSize() const noexcept
{
  return m_current_size;
}

std::size_t AllocationTracker::getHighWatermark() const noexcept
{
  return m_high_watermark;
}

std::size_t AllocationTracker::getActualSize() const noexcept
{
  auto actual_size = m_allocator->getActualSize();
  return actual_size > 0 ? actual_size : m_current_size;
}

std::size_t AllocationTracker::getAllocationCount() const noexcept
{
  return m_allocation_count;
}

Platform AllocationTracker::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits AllocationTracker::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

strategy::AllocationStrategy* AllocationTracker::getAllocationStrategy()
{
  return m_allocator.get();
}

} // namespace strategy
} // namespace umpire
