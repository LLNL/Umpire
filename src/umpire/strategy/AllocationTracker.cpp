//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/AllocationTracker.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

AllocationTracker::AllocationTracker(
  std::unique_ptr<AllocationStrategy>&& allocator) noexcept :
AllocationStrategy(allocator->getName(), allocator->getId()),
mixins::Inspector(),
m_allocator(std::move(allocator))
{
}

void*
AllocationTracker::allocate(std::size_t bytes)
{
  void* ptr = m_allocator->allocate(bytes);

  UMPIRE_LOG(Debug, "Tracking " << ptr << " bytes for " << m_allocator->getName());

  registerAllocation(ptr, bytes, this);

  return ptr;
}

void
AllocationTracker::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "Untracking " << ptr << " bytes for" << m_allocator->getName());

  deregisterAllocation(ptr, this);
  m_allocator->deallocate(ptr);
}

void
AllocationTracker::release()
{
  m_allocator->release();
}

std::size_t
AllocationTracker::getCurrentSize() const noexcept
{
  return m_current_size;
}

std::size_t
AllocationTracker::getHighWatermark() const noexcept
{
  return m_high_watermark;
}

std::size_t
AllocationTracker::getActualSize() const noexcept
{
  return m_allocator->getActualSize();
}

Platform
AllocationTracker::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

strategy::AllocationStrategy*
AllocationTracker::getAllocationStrategy()
{
  return m_allocator.get();
}

} // end of namespace umpire
} // end of namespace strategy
