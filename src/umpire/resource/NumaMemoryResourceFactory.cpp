//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include "umpire/resource/NumaMemoryResourceFactory.hpp"
#include "umpire/resource/NumaMemoryResource.hpp"
#include "umpire/resource/DetectVendor.hpp"

#include "umpire/util/Macros.hpp"

#include <numa.h>

namespace umpire {
namespace resource {

namespace numa {

std::size_t nodeCount() {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");
  return numa_max_possible_node() + 1;
}

std::vector<std::size_t> getHostNodes() {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  std::vector<std::size_t> host_nodes;
  struct bitmask *node_mask = numa_allocate_nodemask();
  struct bitmask *cpus_mask = numa_allocate_cpumask();

  const std::size_t num_nodes = numa_max_possible_node();
  for (std::size_t i = 0; i < num_nodes; i++) {
    if (numa_bitmask_isbitset(node_mask, i)) {
      // Check if this has CPUs
      if (numa_node_to_cpus(i, cpus_mask) != 0) {
        UMPIRE_ERROR("Error getting CPU list for NUMA node.");
      }
      const std::size_t ncpus = numa_bitmask_weight(cpus_mask);
      if (ncpus > 0) {
        // This is a host node
        host_nodes.push_back(i);
      }
    }
  }

  numa_free_nodemask(node_mask);
  numa_free_cpumask(node_mask);

  return host_nodes;
}

std::vector<std::size_t> getDeviceNodes() {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  std::vector<std::size_t> device_nodes;
  struct bitmask *node_mask = numa_allocate_nodemask();
  struct bitmask *cpus_mask = numa_allocate_cpumask();

  const std::size_t num_nodes = numa_max_possible_node();
  for (std::size_t i = 0; i < num_nodes; i++) {
    if (numa_bitmask_isbitset(node_mask, i)) {
      // Check if this has CPUs
      if (numa_node_to_cpus(i, cpus_mask) != 0) {
        UMPIRE_ERROR("Error getting CPU list for NUMA node.");
      }
      const std::size_t ncpus = numa_bitmask_weight(cpus_mask);
      if (ncpus == 0) {
        // This is a host node
        device_nodes.push_back(i);
      }
    }
  }

  numa_free_nodemask(node_mask);
  numa_free_cpumask(node_mask);

  return device_nodes;
}

}

// TODO: Add test that union(host_nodes, device_nodes) == nodes

NumaMemoryResourceFactory::NumaMemoryResourceFactory(const int numa_node_)
  : numa_node(numa_node_) {}

bool
NumaMemoryResourceFactory::isValidMemoryResourceFor(const std::string& UMPIRE_UNUSED_ARG(name),
                                                    const MemoryResourceTraits traits)
  noexcept
{
  return (traits.numa_node == numa_node);
}

std::shared_ptr<MemoryResource>
NumaMemoryResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0;
  traits.numa_node = numa_node;

  traits.vendor = CpuVendorType();
  traits.kind = MemoryResourceTraits::memory_type::DDR;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return std::make_shared<resource::NumaMemoryResource >(id, traits);
}

} // end of namespace resource
} // end of namespace umpire
