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
#include "umpire/util/Numa.hpp"

#include "umpire/util/Macros.hpp"

#include <numa.h>

namespace umpire {
namespace numa {

union AlignedSize {
  std::size_t bytes;
  std::max_align_t a;
};

void* allocate_on_node(const std::size_t bytes, const std::size_t node) {
  AlignedSize* s = static_cast<AlignedSize*>(
    numa_alloc_onnode(sizeof(*s) + bytes, node));
  if (!s) {
    UMPIRE_ERROR("allocate_on_node( " <<
                 "bytes = " << bytes <<
                 ", node = " << node << " ) failed");
  }
  s->bytes = bytes;
  return ++s;
}

void deallocate(void* ptr) {
  AlignedSize *s = static_cast<AlignedSize*>(ptr);
  s--;
  numa_free(s, sizeof(*s) + s->bytes);
}

void* reallocate(void* ptr, const std::size_t new_bytes) {
  AlignedSize *s = static_cast<AlignedSize*>(ptr);
  s--;

  const std::size_t size_aligned = sizeof(*s);

  AlignedSize* new_s = static_cast<AlignedSize*>(
    numa_realloc(s, size_aligned + s->bytes, size_aligned + new_bytes));
  if (!new_s) {
    UMPIRE_ERROR("reallocate( " <<
                 "ptr = " << ptr <<
                 ", new_bytes = " << new_bytes << " ) failed");
  }
  return ++new_s;
}

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

NodeType node_type(const std::size_t node) {
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  struct bitmask* cpus = numa_allocate_cpumask();

  if (numa_node_to_cpus(node, cpus) != 0) UMPIRE_ERROR("An error occured in numa_node_to_cpus()");
  const std::size_t num_cpus = numa_bitmask_weight(cpus);
  numa_free_cpumask(cpus);

  return (num_cpus != 0) ? NodeType::Host : NodeType::Device;
}

} // end namespace numa
} // end namespace umpire

// TODO: Add test that union(host_nodes, device_nodes) == nodes
