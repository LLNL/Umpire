//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

namespace strategy {

#if defined(UMPIRE_V1_DELEGATE_TO_V2)

MonotonicAllocationStrategy::MonotonicAllocationStrategy(const std::string& name, int id, Allocator allocator,
                                                         std::size_t capacity)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "MonotonicAllocationStrategy"},
      m_block{nullptr},
      m_size(0),
      m_capacity(capacity),
      m_allocator(allocator.getAllocationStrategy()),
      m_v1_backed_parent{std::make_unique<detail::v1_backed_memory>(m_allocator)},
      m_delegate{std::make_unique<monotonic_buffer<detail::v1_backed_memory>>(name, m_v1_backed_parent.get(),
                                                                              m_capacity)}
{
  // v1 acquires its single backing block eagerly in the constructor and
  // hands out raw, unaligned bump-pointer offsets into it; v2's
  // monotonic_buffer<Memory> constructor does the same block acquisition
  // (through the v1-backed bridge, so it is tracked in the v2 registry too),
  // so m_block is simply pointed at that same buffer and all subsequent
  // bump-pointer arithmetic below is performed natively exactly as v1 does,
  // rather than through m_delegate->allocate() (whose alignment and
  // zero-byte-request semantics differ from v1's, see monotonic_buffer.hpp).
  m_block = m_delegate->get_buffer();
}

MonotonicAllocationStrategy::~MonotonicAllocationStrategy()
{
  // m_delegate's own destructor returns the single backing block to the
  // v1-backed parent; no separate deallocate_internal() call is needed here.
}

void* MonotonicAllocationStrategy::allocate(std::size_t bytes)
{
  void* ret = static_cast<char*>(m_block) + m_size;
  m_size += bytes;

  if (m_size > m_capacity) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("MonotonicAllocationStrategy capacity exceeded {} > {}", m_size, m_capacity));
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void MonotonicAllocationStrategy::deallocate(void* UMPIRE_UNUSED_ARG(ptr), std::size_t UMPIRE_UNUSED_ARG(size))
{
}

std::size_t MonotonicAllocationStrategy::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_size);
  return m_size;
}

std::size_t MonotonicAllocationStrategy::getHighWatermark() const noexcept
{
  // Kept native (v1's quirk): always returns the constant capacity rather
  // than a real observed peak, unlike v2's get_high_watermark(). This is
  // deliberately NOT delegated to m_delegate->get_high_watermark().
  UMPIRE_LOG(Debug, "() returning " << m_capacity);
  return m_capacity;
}

Platform MonotonicAllocationStrategy::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits MonotonicAllocationStrategy::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

#else // !defined(UMPIRE_V1_DELEGATE_TO_V2)

MonotonicAllocationStrategy::MonotonicAllocationStrategy(const std::string& name, int id, Allocator allocator,
                                                         std::size_t capacity)
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "MonotonicAllocationStrategy"},
      m_size(0),
      m_capacity(capacity),
      m_allocator(allocator.getAllocationStrategy())
{
  m_block = m_allocator->allocate_internal(m_capacity);
}

MonotonicAllocationStrategy::~MonotonicAllocationStrategy()
{
  m_allocator->deallocate_internal(m_block, m_capacity);
}

void* MonotonicAllocationStrategy::allocate(std::size_t bytes)
{
  void* ret = static_cast<char*>(m_block) + m_size;
  m_size += bytes;

  if (m_size > m_capacity) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("MonotonicAllocationStrategy capacity exceeded {} > {}", m_size, m_capacity));
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void MonotonicAllocationStrategy::deallocate(void* UMPIRE_UNUSED_ARG(ptr), std::size_t UMPIRE_UNUSED_ARG(size))
{
}

std::size_t MonotonicAllocationStrategy::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_size);
  return m_size;
}

std::size_t MonotonicAllocationStrategy::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_capacity);
  return m_capacity;
}

Platform MonotonicAllocationStrategy::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits MonotonicAllocationStrategy::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

#endif // defined(UMPIRE_V1_DELEGATE_TO_V2)

} // end of namespace strategy
} // end of namespace umpire
