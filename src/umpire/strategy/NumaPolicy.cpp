//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/NumaPolicy.hpp"

#include <algorithm>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/numa.hpp"

namespace umpire {

namespace strategy {

NumaPolicy::NumaPolicy(const std::string& name, int id, Allocator allocator,
                       int numa_node)
    : AllocationStrategy(name, id),
      m_allocator(allocator.getAllocationStrategy()),
      m_platform(Platform::host),
      m_node(numa_node)
{
  if (numa_node < 0) {
    UMPIRE_ERROR("NumaPolicy error: NUMA nodes are always non-negative ints");
  }
  if (allocator.getPlatform() != Platform::host) {
    UMPIRE_ERROR("NumaPolicy error: allocator is not of cpu type");
  }

#if defined(UMPIRE_ENABLE_DEVICE)
  auto host_nodes = numa::get_host_nodes();
  if (std::find(host_nodes.begin(), host_nodes.end(), m_node) ==
      host_nodes.end()) {
    // This is a device node
#if defined(UMPIRE_ENABLE_CUDA)
    m_platform = Platform::cuda;
#elif defined(UMPIRE_ENABLE_HIP)
    m_platform = Platform::hip;
#else
    UMPIRE_ERROR("Could not determine device platform.");
#endif
  }
#endif
}

void* NumaPolicy::allocate(std::size_t bytes)
{
  void* ret = m_allocator->allocate(bytes);

  numa::move_to_node(ret, bytes, m_node);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

  return ret;
}

void NumaPolicy::deallocate(void* ptr)
{
  m_allocator->deallocate(ptr);
}

Platform NumaPolicy::getPlatform() noexcept
{
  return m_platform;
}

MemoryResourceTraits NumaPolicy::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

int NumaPolicy::getNode() const noexcept
{
  return m_node;
}

} // end of namespace strategy
} // end of namespace umpire
