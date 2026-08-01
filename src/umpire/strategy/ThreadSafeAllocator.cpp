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

#if defined(UMPIRE_V1_DELEGATE_TO_V2)

// Delegated implementation: minimal-diff (see SizeLimiter.cpp for shared
// rationale). allocate()/deallocate() forward through a v2
// thread_safe<v1_backed_memory>, which serializes calls to the wrapped
// parent's allocate()/deallocate() (here, the v1 parent via
// v1_backed_memory) with its own std::mutex. The v1 counter path in
// AllocationStrategy::allocate_internal/deallocate_internal is untouched.
ThreadSafeAllocator::ThreadSafeAllocator(const std::string& name, int id, Allocator allocator)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "ThreadSafeAllocator"},
      m_allocator(allocator.getAllocationStrategy()),
      m_mutex(),
      m_v1_backed_parent{std::make_unique<detail::v1_backed_memory>(m_allocator)},
      m_delegate{std::make_unique<thread_safe<detail::v1_backed_memory>>(name, m_v1_backed_parent.get())}
{
}

void* ThreadSafeAllocator::allocate(std::size_t bytes)
{
  return m_delegate->allocate(bytes);
}

void ThreadSafeAllocator::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  m_delegate->deallocate(ptr);
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

#else // !defined(UMPIRE_V1_DELEGATE_TO_V2)

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

#endif // UMPIRE_V1_DELEGATE_TO_V2

} // end of namespace strategy
} // end of namespace umpire
