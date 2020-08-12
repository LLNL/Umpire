//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/ThreadSafeAllocator.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

ThreadSafeAllocator::ThreadSafeAllocator(const std::string& name, int id,
                                         Allocator allocator)
    : AllocationStrategy(name, id),
      m_allocator(allocator.getAllocationStrategy()),
      m_mutex()
{
}

void* ThreadSafeAllocator::allocate(std::size_t bytes)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  void* ret = m_allocator->allocate(bytes);
  return ret;
}

void ThreadSafeAllocator::deallocate(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_allocator->deallocate(ptr);
}

Platform ThreadSafeAllocator::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits ThreadSafeAllocator::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire
