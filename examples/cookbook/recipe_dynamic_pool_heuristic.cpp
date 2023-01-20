//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/wrap_allocator.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  // _sphinx_tag_tut_creat_heuristic_fun_start
  //
  // Create a heuristic function that will return true to the DynamicPoolList
  // object when the threshold of releasable size to total size is 75%.
  //
  auto heuristic_function = umpire::strategy::DynamicPoolList::percent_releasable(75);
  // _sphinx_tag_tut_creat_heuristic_fun_end

  // _sphinx_tag_tut_use_heuristic_fun_start
  //
  // Create a pool with an initial block size of 1 Kb and 1 Kb block size for
  // all subsequent allocations and with our previously created heuristic
  // function.
  //
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPoolList>("HOST_POOL", allocator, 1024ul, 1024ul,
                                                                              16, heuristic_function);
  // _sphinx_tag_tut_use_heuristic_fun_end

  //
  // Obtain a pointer to our specific DynamicPoolList instance in order to see the
  // DynamicPoolList-specific statistics
  //
  auto dynamic_pool = umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolList>(pooled_allocator);

  void* a[4];
  for (int i = 0; i < 4; ++i)
    a[i] = pooled_allocator.allocate(1024);

  for (int i = 0; i < 4; ++i) {
    pooled_allocator.deallocate(a[i]);
    std::cout << "Pool has " << pooled_allocator.getActualSize() << " bytes of memory. "
              << pooled_allocator.getCurrentSize() << " bytes are used. " << dynamic_pool->getBlocksInPool()
              << " blocks are in the pool. " << dynamic_pool->getReleasableSize() << " bytes are releaseable. "
              << std::endl;
  }

  return 0;
}
