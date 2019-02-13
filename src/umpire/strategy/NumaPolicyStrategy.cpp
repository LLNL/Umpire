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
#include "umpire/strategy/NumaPolicyStrategy.hpp"
#include "umpire/util/Macros.hpp"

#include "umpire/ResourceManager.hpp"

#include <numa.h>
#include <numaif.h>

namespace umpire {

namespace strategy {

NumaPolicyStrategy::NumaPolicyStrategy(
    const std::string& name,
    int id,
    int numa_node,
    Allocator allocator) :
  AllocationStrategy(name, id),
  m_node(numa_node),
  m_mask(numa_bitmask_alloc(numa_max_node() + 1)),
  m_allocator(allocator.getAllocationStrategy())
{
  numa_bitmask_clearall(m_mask);
  numa_bitmask_setbit(m_mask, m_node);
}

NumaPolicyStrategy::~NumaPolicyStrategy()
{
  numa_bitmask_free(m_mask);
}

void*
NumaPolicyStrategy::allocate(size_t bytes)
{
  void *ret = m_allocator->allocate(bytes);

  if (mbind(ret, bytes, MPOL_BIND, m_mask->maskp, m_mask->size + 1, MPOL_MF_STRICT) != 0) {
    UMPIRE_ERROR("NumaPolicyStrategy mbind( ret = " << ret << ", bytes = " << bytes << ", node = " << m_node << " ) failed");
  }

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void
NumaPolicyStrategy::deallocate(void* ptr)
{
  m_allocator->deallocate(ptr);
}

long
NumaPolicyStrategy::getCurrentSize() const noexcept
{
  return m_allocator->getCurrentSize();
}

long
NumaPolicyStrategy::getHighWatermark() const noexcept
{
  return m_allocator->getHighWatermark();
}

Platform
NumaPolicyStrategy::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
