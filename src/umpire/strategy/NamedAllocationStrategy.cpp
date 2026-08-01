//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/NamedAllocationStrategy.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

#if defined(UMPIRE_V1_DELEGATE_TO_V2)

// Delegated implementation: minimal-diff (see SizeLimiter.cpp for rationale
// shared across all three converted strategies). The v1 counter path in
// AllocationStrategy::allocate_internal/deallocate_internal is untouched;
// only the private allocate()/deallocate() virtuals forward through a v2
// named<v1_backed_memory>, which is itself a pure passthrough decorator (see
// include/umpire/strategy/named.hpp) so behavior is unchanged.
NamedAllocationStrategy::NamedAllocationStrategy(const std::string& name, int id, Allocator allocator)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "NamedAllocationStrategy"},
      m_allocator(allocator.getAllocationStrategy()),
      m_v1_backed_parent{std::make_unique<detail::v1_backed_memory>(m_allocator)},
      m_delegate{std::make_unique<named<detail::v1_backed_memory>>(name, m_v1_backed_parent.get())}
{
}

void* NamedAllocationStrategy::allocate(std::size_t bytes)
{
  return m_delegate->allocate(bytes);
}

void* NamedAllocationStrategy::allocate_named(const std::string& name, std::size_t bytes)
{
  // named<Memory> has no per-allocation naming concept; continue calling the
  // v1 parent directly, same as the native (non-delegated) implementation.
  return m_allocator->allocate_named_internal(name, bytes);
}

void NamedAllocationStrategy::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  m_delegate->deallocate(ptr);
}

Platform NamedAllocationStrategy::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits NamedAllocationStrategy::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

#else // !defined(UMPIRE_V1_DELEGATE_TO_V2)

NamedAllocationStrategy::NamedAllocationStrategy(const std::string& name, int id, Allocator allocator)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "NamedAllocationStrategy"},
      m_allocator(allocator.getAllocationStrategy())
{
}

void* NamedAllocationStrategy::allocate(std::size_t bytes)
{
  return m_allocator->allocate_internal(bytes);
}

void* NamedAllocationStrategy::allocate_named(const std::string& name, std::size_t bytes)
{
  return m_allocator->allocate_named_internal(name, bytes);
}

void NamedAllocationStrategy::deallocate(void* ptr, std::size_t size)
{
  return m_allocator->deallocate_internal(ptr, size);
}

Platform NamedAllocationStrategy::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits NamedAllocationStrategy::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

#endif // UMPIRE_V1_DELEGATE_TO_V2

} // end of namespace strategy
} // end of namespace umpire
