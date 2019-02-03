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
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

#include <iostream>

int main(int, char**) {
#if 0
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  /*
   * Create a heuristic function that will return true to the DynamicPool
   * object when the threshold of releasable size to total size is 50%.
   */
  umpire::strategy::DynamicPool::Coalesce_Heuristic heuristic_function =
                        umpire::strategy::heuristic_percent_releasable(50);

  /* 
   * Create a pool with an initial block size of 1 Kb and 1 Kb block size for
   * all subsequent allocations and with our previously created heuristic
   * function.
   */
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
                             "HOST_POOL", allocator
                            , 1024ul, 1024ul
                            , heuristic_function); 

  auto strategy = pooled_allocator.getAllocationStrategy();
  auto tracker = std::dynamic_pointer_cast<umpire::strategy::AllocationTracker>(strategy);

  if (tracker) {
    strategy = tracker->getAllocationStrategy();
  }

  auto dynamic_pool = std::dynamic_pointer_cast<umpire::strategy::DynamicPool>(strategy);

  void* a1 = pooled_allocator.allocate(1024);
  void* a2 = pooled_allocator.allocate(1024);
  void* a3 = pooled_allocator.allocate(1024);
  void* a4 = pooled_allocator.allocate(1024);

  std::cout << "Pool has allocated " << pooled_allocator.getActualSize()
            << " bytes of memory. " << pooled_allocator.getCurrentSize()
            << " bytes are used." << dynamic_pool->getBlocksInPool()
            << " blocks are in pool." << dynamic_pool->getReleasableSize()
            << " bytes are releaseable." << std::endl;

  //
  // This will cause on only one block to be free which will show 0 bytes as
  // releasable since we need > 1 block to coalesce.
  pooled_allocator.deallocate(a1);
                          
  std::cout << "Pool has allocated " << pooled_allocator.getActualSize()
            << " bytes of memory. " << pooled_allocator.getCurrentSize()
            << " bytes are used." << dynamic_pool->getBlocksInPool()
            << " blocks are in pool." << dynamic_pool->getReleasableSize()
            << " bytes are releaseable." << std::endl;


  /*
   * Grow pool to ~12 by grabbing a 8Gb chunk
   */
  void* grow = pooled_allocator.allocate( 8ul * 1024ul * 1024ul * 1024ul );
  pooled_allocator.deallocate(grow);

  std::cout << "Pool has allocated " << pooled_allocator.getActualSize()
            << " bytes of memory. " << pooled_allocator.getCurrentSize()
            << " bytes are used" << std::endl;

  /*
   * Shrink pool back to ~4Gb
   */
  pooled_allocator.release();
  std::cout << "Pool has allocated " << pooled_allocator.getActualSize() 
            << " bytes of memory. " << pooled_allocator.getCurrentSize()
            << " bytes are used" << std::endl;

#endif
  return 0;
}
