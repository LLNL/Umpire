//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

ThreadSafeAllocator::ThreadSafeAllocator(
    const std::string& name,
    int id,
    Allocator allocator) :
  AllocationStrategy(name, id),
  m_allocator(allocator.getAllocationStrategy()),
  m_mutex()
{
}

void*
ThreadSafeAllocator::allocate(std::size_t bytes)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  void *ret = m_allocator->allocate(bytes);
  return ret;
}

void
ThreadSafeAllocator::deallocate(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_allocator->deallocate(ptr);
}

std::size_t
ThreadSafeAllocator::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t
ThreadSafeAllocator::getHighWatermark() const noexcept
{
  return 0;
}

Platform
ThreadSafeAllocator::getPlatform() noexcept
{

  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
