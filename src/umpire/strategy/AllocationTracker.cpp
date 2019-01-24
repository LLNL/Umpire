//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

namespace umpire {
namespace strategy {

AllocationTracker::AllocationTracker(
  const std::string& name,
  int id,
  Allocator allocator) noexcept :
AllocationStrategy(name, id),
mixins::Inspector(),
m_allocator(allocator.getAllocationStrategy())
{
}

void*
AllocationTracker::allocate(size_t bytes)
{
  void* ptr = m_allocator->allocate(bytes);

  registerAllocation(ptr, bytes, this->shared_from_this());

  return ptr;
}

void
AllocationTracker::deallocate(void* ptr)
{
  deregisterAllocation(ptr);
  m_allocator->deallocate(ptr);
}

long
AllocationTracker::getCurrentSize() const noexcept
{
  return m_current_size;
}

long
AllocationTracker::getHighWatermark() const noexcept
{
  return m_high_watermark;
}

long
AllocationTracker::getActualSize() const noexcept
{
  return m_allocator->getActualSize();
}

Platform
AllocationTracker::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

std::shared_ptr<umpire::strategy::AllocationStrategy> 
AllocationTracker::getAllocationStrategy()
{
  return m_allocator;
}

} // end of namespace umpire
} // end of namespace strategy
