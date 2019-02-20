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
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/NumaPolicy.hpp"

#include "umpire/util/numa.hpp"
#include "umpire/util/Macros.hpp"

#include <iostream>

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();

  const std::size_t alloc_size = 5 * umpire::get_page_size();

  // Get a list of the host NUMA nodes (e.g. one per socket)
  auto host_nodes = umpire::numa::get_host_nodes();

  if (host_nodes.size() < 1) {
    UMPIRE_ERROR("No NUMA nodes detected");
  }

  // Create an allocator on the first NUMA node
  auto host_src_alloc = rm.makeAllocator<umpire::strategy::NumaPolicy>(
    "host_numa_src_alloc", host_nodes[0], rm.getAllocator("HOST"));

  // Create an allocation on that node
  void* src_ptr = host_src_alloc.allocate(alloc_size);

#if !defined(UMPIRE_ENABLE_CUDA)
  if (host_nodes.size() > 1) {
    // Create an allocator on another host NUMA node.
    auto host_dst_alloc = rm.makeAllocator<umpire::strategy::NumaPolicy>(
      "host_numa_dst_alloc", host_nodes[1], rm.getAllocator("HOST"));

    // Move the memory
    void* dst_ptr = rm.move(src_ptr, host_dst_alloc);

    // The pointer shouldn't change even though the memory location changes
    if (dst_ptr != src_ptr) {
      UMPIRE_ERROR("Pointers should match");
    }

    // Touch it
    rm.memset(dst_ptr, 0, alloc_size);

    if (umpire::numa::get_location(dst_ptr) != host_nodes[0]) {
      UMPIRE_ERROR("Move was unsuccessful");
    }
  }
#else
  // Get a list of the device nodes
  auto device_nodes = umpire::numa::get_device_nodes();

  if (device_nodes.size() > 0) {
    // Create an allocator on the first device NUMA node. Note that
    // this still requires using the "HOST" allocator. The allocations
    // are moved after the address space is reserved.
    auto device_alloc = rm.makeAllocator<umpire::strategy::NumaPolicy>(
      "device_numa_src_alloc", device_nodes[0], rm.getAllocator("HOST"));

    // Move the memory
    void* dst_ptr = rm.move(src_ptr, device_alloc);

    // The pointer shouldn't change even though the memory location changes
    if (dst_ptr != src_ptr) {
      UMPIRE_ERROR("Pointers should match");
    }

    // Touch it
    rm.memset(dst_ptr, 0, alloc_size);

    if (umpire::numa::get_location(dst_ptr) != device_nodes[0]) {
      UMPIRE_ERROR("Move was unsuccessful");
    }
  }
#endif

  // Clean up by deallocating from the original allocator, since the
  // allocation record is still associated with that allocator
  host_src_alloc.deallocate(src_ptr);

  return 0;
}
