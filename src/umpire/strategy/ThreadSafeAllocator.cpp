//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/ThreadSafeAllocator.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

ThreadSafeAllocator::ThreadSafeAllocator(const std::string& name, int id, Allocator allocator)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "ThreadSafeAllocator"},
      m_allocator(allocator.getAllocationStrategy()),
      m_mutex()
{
}

void* ThreadSafeAllocator::allocate(std::size_t bytes)
{
  return m_allocator->allocate_internal(bytes);
}

void ThreadSafeAllocator::deallocate(void* ptr, std::size_t size)
{
  m_allocator->deallocate_internal(ptr, size);
}

Platform ThreadSafeAllocator::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits ThreadSafeAllocator::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

std::mutex* ThreadSafeAllocator::get_mutex()
{
  return &m_mutex;
}

} // end of namespace strategy
} // end of namespace umpire
