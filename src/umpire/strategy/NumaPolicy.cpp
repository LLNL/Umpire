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
#include "umpire/strategy/NumaPolicy.hpp"
#include "umpire/util/Macros.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Numa.hpp"

namespace umpire {

namespace strategy {

NumaPolicy::NumaPolicy(
    const std::string& name,
    int id,
    int numa_node,
    Allocator allocator) :
  AllocationStrategy(name, id),
  m_node(numa_node),
  m_allocator(allocator.getAllocationStrategy())
{
  if (allocator.getPlatform() != Platform::cpu) {
    UMPIRE_ERROR("NumaPolicy error: allocator is not of cpu type");
  }
}

void*
NumaPolicy::allocate(size_t bytes)
{
  void *ret = m_allocator->allocate(bytes);

  numa::move_to_node(ret, bytes, m_node);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void
NumaPolicy::deallocate(void* ptr)
{
  m_allocator->deallocate(ptr);
}

long
NumaPolicy::getCurrentSize() const noexcept
{
  return m_allocator->getCurrentSize();
}

long
NumaPolicy::getHighWatermark() const noexcept
{
  return m_allocator->getHighWatermark();
}

Platform
NumaPolicy::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

int
NumaPolicy::getNode() const noexcept
{
  return m_node;
}

} // end of namespace strategy
} // end of namespace umpire
