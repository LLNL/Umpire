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
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

#include <iostream>

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  //
  // Create a heuristic function that will return true to the DynamicPool
  // object when the threshold of releasable size to total size is 75%.
  //
  auto heuristic_function = umpire::strategy::heuristic_percent_releasable(75);

  //
  // Create a pool with an initial block size of 1 Kb and 1 Kb block size for
  // all subsequent allocations and with our previously created heuristic
  // function.
  //
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
                             "HOST_POOL"
                            , allocator
                            , 1024ul
                            , 1024ul
                            , heuristic_function);

  //
  // Obtain a pointer to our specifi DynamicPool instance in order to see the
  // DynamicPool-specific statistics
  //
  auto strategy = pooled_allocator.getAllocationStrategy();
  auto tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(strategy);

  if (tracker)
    strategy = tracker->getAllocationStrategy();

  auto dynamic_pool = dynamic_cast<umpire::strategy::DynamicPool*>(strategy);

  void* a[4];
  for (int i = 0; i < 4; ++i)
    a[i] = pooled_allocator.allocate(1024);

  for (int i = 0; i < 4; ++i) {
    pooled_allocator.deallocate(a[i]);
    std::cout
      << "Pool has " << pooled_allocator.getActualSize() << " bytes of memory. "
      << pooled_allocator.getCurrentSize() << " bytes are used. "
      << dynamic_pool->getBlocksInPool() << " blocks are in the pool. "
      << dynamic_pool->getReleasableSize() << " bytes are releaseable. "
      << std::endl;
  }

  return 0;
}
