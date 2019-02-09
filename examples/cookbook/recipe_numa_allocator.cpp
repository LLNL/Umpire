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
#include "umpire/strategy/AllocationAdvisor.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/util/Numa.hpp"

#include "umpire/util/Exception.hpp"

#include <iostream>

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();

  /*
   * Get the preferred NUMA node for allocation.
   */
  const int node = numa::preferred_node();

  /*
   * Create a traits object with the numa node id and pass to
   * getAllocatorFor.
   */
  umpire::resource::MemoryResourceTraits traits;
  traits.numa_node = node;

  auto allocator = rm.getAllocatorFor(traits);

  // Create a 4MB allocation
  const std::size_t size = 4*1024*1024;
  void *ptr = allocator.allocate(size);

  // Touch the memory and release it
  rm.memset(ptr, 0, size);
  allocator.deallocate(ptr);

  return 0;
}
