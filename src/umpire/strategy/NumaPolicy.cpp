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

#include "umpire/util/numa.hpp"

#include <algorithm>

namespace umpire {

namespace strategy {

NumaPolicy::NumaPolicy(
    const std::string& name,
    int id,
    int numa_node,
    Allocator allocator) :
  AllocationStrategy(name, id),
  m_allocator(allocator.getAllocationStrategy()),
  m_platform(Platform::cpu),
  m_node(numa_node)
{
  if (numa_node < 0) {
    UMPIRE_ERROR("NumaPolicy error: NUMA nodes are always non-negative ints");
  }
  if (allocator.getPlatform() != Platform::cpu) {
    UMPIRE_ERROR("NumaPolicy error: allocator is not of cpu type");
  }

#if defined(UMPIRE_ENABLE_DEVICE)
  auto host_nodes = numa::get_host_nodes();
  if (std::find(host_nodes.begin(), host_nodes.end(),
                m_node) == host_nodes.end()) {
    // This is a device node
    // TODO: Could both these be enabled? If so, find a way to
    // distinguish these at run-time.
#if defined(UMPIRE_ENABLE_CUDA)
    m_platform = Platform::cuda;
#elif defined(UPIRE_ENABLE_ROCM)
    m_platform = Platform::rocm;
#else
    UMPIRE_ERROR("Could not determine device platform.");
#endif
  }
#endif
}

void
NumaPolicy::finalize()
{
  m_allocator->finalize();
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
  return m_platform;
}

int
NumaPolicy::getNode() const noexcept
{
  return m_node;
}

} // end of namespace strategy
} // end of namespace umpire
