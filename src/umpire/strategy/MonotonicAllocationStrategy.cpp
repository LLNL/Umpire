//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

namespace strategy {

MonotonicAllocationStrategy::MonotonicAllocationStrategy(
    const std::string& name, int id, Allocator allocator, std::size_t capacity)
    : AllocationStrategy(name, id),
      m_size(0),
      m_capacity(capacity),
      m_allocator(allocator.getAllocationStrategy())
{
  m_block = m_allocator->allocate(m_capacity);
}

MonotonicAllocationStrategy::~MonotonicAllocationStrategy()
{
  m_allocator->deallocate(m_block);
}

void* MonotonicAllocationStrategy::allocate(std::size_t bytes)
{
  void* ret = static_cast<char*>(m_block) + m_size;
  m_size += bytes;

  if (m_size > m_capacity) {
    UMPIRE_ERROR("MonotonicAllocationStrategy capacity exceeded "
                 << m_size << " > " << m_capacity);
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void MonotonicAllocationStrategy::deallocate(void* UMPIRE_UNUSED_ARG(ptr))
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

} // end of namespace strategy
} // end of namespace umpire
