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
#include "umpire/util/numa.hpp"

#include "umpire/util/Macros.hpp"

#include <numa.h>
#include <numaif.h>
#include <unistd.h>

namespace umpire {

long s_cpu_page_size = sysconf(_SC_PAGESIZE);

namespace numa {

int preferred_node()
{
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");
  return numa_preferred();
}

void move_to_node(void* ptr, size_t bytes, int node)
{
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  struct bitmask* mask = numa_bitmask_alloc(numa_max_node() + 1);
  numa_bitmask_clearall(mask);
  numa_bitmask_setbit(mask, node);

  if (mbind(ptr, bytes, MPOL_BIND, mask->maskp, mask->size + 1, MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    UMPIRE_ERROR("numa::move_to_node error: mbind( ptr = " << ptr <<
                 ", bytes = " << bytes <<
                 ", node = " << node << " ) failed");
  }

  numa_bitmask_free(mask);
}

int get_location(void* ptr)
{
  int numa_node = -1;
  if (get_mempolicy(&numa_node, NULL, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR) != 0) {
    UMPIRE_ERROR("numa::get_location error: get_mempolicy( ptr = " << ptr << ") failed");
  }
  return numa_node;
}

std::vector<int> get_host_nodes()
{
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  std::vector<int> host_nodes;
  struct bitmask* cpus = numa_allocate_cpumask();

  const int size = numa_all_nodes_ptr->size;
  for (int i = 0; i < size; i++) {
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

std::vector<int> get_device_nodes()
{
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  std::vector<int> device_nodes;
  struct bitmask* cpus = numa_allocate_cpumask();

  const int size = numa_all_nodes_ptr->size;
  for (int i = 0; i < size; i++) {
    if (numa_bitmask_isbitset(numa_all_nodes_ptr, i)) {

      // Check if this has CPUs
      if (numa_node_to_cpus(i, cpus) != 0) {
        UMPIRE_ERROR("Error getting CPU list for NUMA node.");
      }

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

std::vector<int> get_allocatable_nodes()
{
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");

  struct bitmask* mask = numa_get_mems_allowed();
  std::vector<int> nodes;
  const int size = mask->size;
  for (int i = 0; i < size; i++) {
    if (numa_bitmask_isbitset(mask, i)) nodes.push_back(i);
  }

  numa_free_nodemask(mask);
  return nodes;
}


} // end namespace numa
} // end namespace umpire
