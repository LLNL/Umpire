//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/SizeLimiter.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/strategy/detail/v1_backed_memory.hpp"
#include "umpire/strategy/size_limiter.hpp"
#endif

namespace umpire {
namespace strategy {

#if defined(UMPIRE_V1_DELEGATE_TO_V2)

// Delegated implementation: the private allocate()/deallocate() virtuals
// forward to a v2 strategy::size_limiter<...> composed over a
// strategy::detail::v1_backed_memory bridge wrapping the v1 parent. This is
// the MINIMAL-diff approach discussed in the mission background: the v1
// AllocationStrategy::allocate_internal()/deallocate_internal() counter path
// (m_current_size/m_high_watermark/m_allocation_count) is untouched -- it
// still runs exactly as before via the outer Allocator.inl call sequence --
// so getCurrentSize()/getHighWatermark()/getAllocationCount() behavior and
// numbers are byte-for-byte identical to the native path. Only the *actual*
// bytes-tracked-for-limiting/threading/naming logic is delegated to v2.
//
// The v1 exception contract is preserved exactly: v2's size_limiter throws
// umpire::logic_error on limit-exceeded, but v1 callers assert
// umpire::out_of_memory_error (see tests/integration/strategy_tests.cpp,
// TEST(SizeLimiter, Host)) with the message "Size limit exceeded.". Rather
// than changing v1's public exception contract, catch the v2 exception here
// and re-throw the identical v1 exception type/message.
SizeLimiter::SizeLimiter(const std::string& name, int id, Allocator allocator, std::size_t size_limit)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "SizeLimiter"},
      m_allocator(allocator.getAllocationStrategy()),
      m_size_limit(size_limit),
      m_total_size(0),
      m_v1_backed_parent{std::make_unique<detail::v1_backed_memory>(m_allocator)},
      m_delegate{std::make_unique<size_limiter<detail::v1_backed_memory>>(name, m_v1_backed_parent.get(), size_limit)}
{
}

void* SizeLimiter::allocate(std::size_t bytes)
{
  try {
    return m_delegate->allocate(bytes);
  } catch (const umpire::logic_error&) {
    UMPIRE_ERROR(out_of_memory_error, "Size limit exceeded.");
  }
}

void SizeLimiter::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  m_delegate->deallocate(ptr);
}

Platform SizeLimiter::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits SizeLimiter::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

#else // !defined(UMPIRE_V1_DELEGATE_TO_V2)

SizeLimiter::SizeLimiter(const std::string& name, int id, Allocator allocator, std::size_t size_limit)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "SizeLimiter"},
      m_allocator(allocator.getAllocationStrategy()),
      m_size_limit(size_limit),
      m_total_size(0)
{
}

void* SizeLimiter::allocate(std::size_t bytes)
{
  m_total_size += bytes;

  if (m_total_size > m_size_limit) {
    m_total_size -= bytes;
    UMPIRE_ERROR(out_of_memory_error, "Size limit exceeded.");
  }

  return m_allocator->allocate_internal(bytes);
}

void SizeLimiter::deallocate(void* ptr, std::size_t size)
{
  m_total_size -= size;
  m_allocator->deallocate_internal(ptr, size);
}

Platform SizeLimiter::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits SizeLimiter::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

#endif // UMPIRE_V1_DELEGATE_TO_V2

} // end of namespace strategy
} // end of namespace umpire
