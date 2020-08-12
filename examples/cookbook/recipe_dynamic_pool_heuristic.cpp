//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/util/Macros.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  //
  // Create a heuristic function that will return true to the DynamicPool
  // object when the threshold of releasable size to total size is 75%.
  //
  auto heuristic_function =
      umpire::strategy::DynamicPool::percent_releasable(75);

  //
  // Create a pool with an initial block size of 1 Kb and 1 Kb block size for
  // all subsequent allocations and with our previously created heuristic
  // function.
  //
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "HOST_POOL", allocator, 1024ul, 1024ul, 16, heuristic_function);

  //
  // Obtain a pointer to our specifi DynamicPool instance in order to see the
  // DynamicPool-specific statistics
  //
  auto dynamic_pool =
      umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(
          pooled_allocator);

  void* a[4];
  for (int i = 0; i < 4; ++i)
    a[i] = pooled_allocator.allocate(1024);

  for (int i = 0; i < 4; ++i) {
    pooled_allocator.deallocate(a[i]);
    std::cout << "Pool has " << pooled_allocator.getActualSize()
              << " bytes of memory. " << pooled_allocator.getCurrentSize()
              << " bytes are used. " << dynamic_pool->getBlocksInPool()
              << " blocks are in the pool. "
              << dynamic_pool->getReleasableSize() << " bytes are releaseable. "
              << std::endl;
  }

  return 0;
}
