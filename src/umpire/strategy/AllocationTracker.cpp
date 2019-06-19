//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/AllocationTracker.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

AllocationTracker::AllocationTracker(
  const std::string& name,
  int id,
  std::unique_ptr<AllocationStrategy>&& allocator) noexcept :
AllocationStrategy(name, id),
mixins::Inspector(),
m_allocator(std::move(allocator))
{
}

void*
AllocationTracker::allocate(std::size_t bytes)
{
  void* ptr = m_allocator->allocate(bytes);

  registerAllocation(ptr, bytes, this);

  return ptr;
}

void
AllocationTracker::deallocate(void* ptr)
{
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
