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

#include "umpire/strategy/SICMStrategy.hpp"

#include "umpire/util/numa.hpp"
#include "umpire/util/Macros.hpp"

extern "C" {
#include "sicm_low.h"
}

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();

  const std::size_t alloc_size = 5 * umpire::get_page_size();

  // Get the list of devices recognized by SICM
  sicm_device_list devs = sicm_init();

  if ((devs.count / 3) < 1) { // each NUMA node creates 3 SICM devices
    UMPIRE_ERROR("SICM did not detect any devices");
  }

  // Create an allocator on the first SICM device
  auto sicm_src_alloc = rm.makeAllocator<umpire::strategy::SICMStrategy>(
    "sicm_src_alloc", 0);

  // Create an allocation on that device
  void* src_ptr = sicm_src_alloc.allocate(alloc_size);

  // Create another allocation on that device
  void* src_ptr2 = sicm_src_alloc.allocate(alloc_size);

  if ((devs.count / 3) > 1) {                    // want at least 2 NUMA nodes
    const unsigned int dst_dev = devs.count - 3; // destination page size must be the same, so pick device + offset 0 on the NUMA node
    const int dst_node = devs.devices[dst_dev].node;

    // Create an allocator on another SICM device
    auto sicm_dst_alloc = rm.makeAllocator<umpire::strategy::SICMStrategy>(
        "sicm_dst_alloc", dst_dev);

    // Move the entire arena
    void* dst_ptr = rm.move(src_ptr, sicm_dst_alloc);

    // The pointer shouldn't change even though the memory location changes
    if (dst_ptr != src_ptr) {
      UMPIRE_ERROR("Pointers should match " << dst_ptr << " " << src_ptr);
    }

    // Verify the location of the pointer is on the destination SICM device
    {
      rm.memset(dst_ptr, 0);

      const int new_location = umpire::numa::get_location(dst_ptr);
      if (new_location != dst_node) {
        UMPIRE_ERROR("Move was unsuccessful. Expected location: " << dst_node << " Actual Location: " << new_location);
      }
    }

    // The other pointer moved as well!
    {
      rm.memset(src_ptr2, 0);

      const int new_location = umpire::numa::get_location(src_ptr2);
      if (new_location != dst_node) {
        UMPIRE_ERROR("Move was unsuccessful. Expected location: " << dst_node << " Actual Location: " << new_location);
      }
    }
  }

  // Clean up by deallocating from the original allocator, since the
  // allocation record is still associated with that allocator
  sicm_src_alloc.deallocate(src_ptr);
  sicm_src_alloc.deallocate(src_ptr2);

  sicm_fini();

  return 0;
}
