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
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {

namespace strategy {

MonotonicAllocationStrategy::MonotonicAllocationStrategy(
    const std::string& name,
    int id,
    size_t capacity,
    Allocator allocator) :
  AllocationStrategy(name, id),
  m_size(0),
  m_capacity(capacity),
  m_allocator(allocator.getAllocationStrategy())
{
  m_block = m_allocator->allocate(m_capacity);
}

void*
MonotonicAllocationStrategy::allocate(size_t bytes)
{
  void* ret = static_cast<char*>(m_block) + bytes;
  m_size += bytes;

  if (m_size > m_capacity) {
    UMPIRE_ERROR("MonotonicAllocationStrategy capacity exceeded " << m_size << " > " << m_capacity);
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void
MonotonicAllocationStrategy::deallocate(void* UMPIRE_UNUSED_ARG(ptr))
{}

std::size_t
MonotonicAllocationStrategy::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_size);
  return m_size;
}

std::size_t
MonotonicAllocationStrategy::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_capacity);
  return m_capacity;
}

Platform
MonotonicAllocationStrategy::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
