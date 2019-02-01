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
#include "umpire/resource/MemoryResourceTypes.hpp"
#include "umpire/resource/NumaMemoryResourceFactory.hpp"
#include "umpire/resource/NumaMemoryResource.hpp"
#include "umpire/resource/DetectVendor.hpp"

#include "umpire/util/Macros.hpp"

#include <numa.h>

namespace umpire {
namespace resource {

namespace numa {

std::size_t node_count() {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");
  return numa_max_possible_node() + 1;
}

std::vector<std::size_t> get_host_nodes() {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  std::vector<std::size_t> host_nodes;
  struct bitmask *cpus = numa_allocate_cpumask();

  const std::size_t num_nodes = numa_max_possible_node();
  for (std::size_t i = 0; i < num_nodes; i++) {
    if (numa_bitmask_isbitset(numa_all_nodes_ptr, i)) {

      // Check if this has CPUs
      if (numa_node_to_cpus(i, cpus) != 0) {
        UMPIRE_ERROR("Error getting CPU list for NUMA node.");
      }

      const std::size_t ncpus = numa_bitmask_weight(cpus);
      if (ncpus > 0) {
        // This is a host node
        host_nodes.push_back(i);
      }
    }
  }

  numa_free_cpumask(cpus);
  return host_nodes;
}

std::vector<std::size_t> get_device_nodes() {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  std::vector<std::size_t> device_nodes;
  struct bitmask *cpus = numa_allocate_cpumask();

  const std::size_t num_nodes = numa_max_possible_node();
  for (std::size_t i = 0; i < num_nodes; i++) {
    if (numa_bitmask_isbitset(numa_all_nodes_ptr, i)) {

      // Check if this has CPUs
      if (numa_node_to_cpus(i, cpus) != 0)
        UMPIRE_ERROR("Error getting CPU list for NUMA node.");

      const std::size_t ncpus = numa_bitmask_weight(cpus);
      if (ncpus == 0) {
        // This is a device node
        device_nodes.push_back(i);
      }
    }
  }

  numa_free_cpumask(cpus);
  return device_nodes;
}

std::size_t preferred_node() {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  return numa_preferred();
}

ResourceType node_type(const std::size_t node) {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  struct bitmask* cpus = numa_allocate_cpumask();

  if (numa_node_to_cpus(node, cpus) != 0) UMPIRE_ERROR("An error occured in numa_node_to_cpus()");
  const std::size_t num_cpus = numa_bitmask_weight(cpus);
  numa_free_cpumask(cpus);
  std::cout << "weight = " << num_cpus << std::endl;

  return (cpus != 0) ? ResourceType::Host : ResourceType::Device;
}


} // namespace numa

// TODO: Add test that union(host_nodes, device_nodes) == nodes

NumaMemoryResourceFactory::NumaMemoryResourceFactory(const int numa_node_)
  : m_numa_node(numa_node_),
    m_preferred_node(numa::preferred_node()),
    m_node_type(numa::node_type(m_numa_node))
{
}

bool
NumaMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name,
                                                    const MemoryResourceTraits traits)
  noexcept
{
  const bool valid_for_host = ((name.compare(type_to_string(resource::Host)) == 0) &&
                               (m_numa_node == m_preferred_node));
  const bool valid_for_other = ((traits.numa_node == m_numa_node) &&
                                (m_node_type == numa::ResourceType::Host));
  return valid_for_host || valid_for_other;
}

std::shared_ptr<MemoryResource>
NumaMemoryResourceFactory::create(const std::string& name, int id)
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.numa_node = m_numa_node;

  traits.vendor = cpu_vendor_type();

  return std::make_shared<resource::NumaMemoryResource >(name, id, traits);
}

} // end of namespace resource
} // end of namespace umpire
