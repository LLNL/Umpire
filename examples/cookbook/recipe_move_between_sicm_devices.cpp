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

#include <iostream>

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();

  const std::size_t alloc_size = 5 * umpire::get_page_size();

  // Get the list of devices recognized by SICM
  sicm_device_list devs = sicm_init();

  if (devs.count < 1) {
    UMPIRE_ERROR("SICM did not dectect any devices");
  }

  // Create an allocator on the first SICM node
  auto sicm_src_alloc = rm.makeAllocator<umpire::strategy::SICMStrategy>(
    "sicm_src_alloc", 0);

  // Create an allocation on that node
  void* src_ptr = sicm_src_alloc.allocate(alloc_size);

  if (devs.count > 1) {
    // Create an allocator on another host SICM node.
    auto sicm_dst_alloc = rm.makeAllocator<umpire::strategy::SICMStrategy>(
        "sicm_dst_alloc", ((devs.count - 1) / 3) * 3);

    // Move the entire arena
    void* dst_ptr = rm.move(src_ptr, sicm_dst_alloc);

    // The pointer shouldn't change even though the memory location changes
    if (dst_ptr != src_ptr) {
      UMPIRE_ERROR("Pointers should match " << dst_ptr << " " << src_ptr);
    }

    // Touch it
    rm.memset(dst_ptr, 0);

    // Verify SICM device
    if (umpire::numa::get_location(dst_ptr) != (int) ((devs.count - 1) / 3)) {
      UMPIRE_ERROR("Move was unsuccessful " << umpire::numa::get_location(dst_ptr) << " " << (int) ((devs.count - 1) / 3));
    }
  }

  // Clean up by deallocating from the original allocator, since the
  // allocation record is still associated with that allocator
  sicm_src_alloc.deallocate(src_ptr);

  sicm_fini();

  return 0;
}
