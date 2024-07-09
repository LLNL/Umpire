//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/wrap_allocator.hpp"

int main(int, char**)
{
  //Create the instance of the Resource Manager and use it to create an allocator using the DEVICE memory resource.
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("DEVICE");

  /*
   * Set up Percent Releasable and Blocks Releasable heuristics. The Percent Releasable heuristic
   * will coalesce the pool when some percentage of bytes in the pool is releasable (i.e. free).
   * The Blocks Releasable heuristic will coalesce the pool when a certain number of blocks in
   * the pool is releasable (i.e. free). Each heuristic function takes a parameter that specifies
   * either the percentage or the number of blocks, depending on which heuristic it is.
   * Below, do the following:
   * 1. Create a Percent Releasable heuristic function that will coalesce when the entire pool is releasable.
   * 2. Create a Percent Releasable heuristic function that will coalesce when 75% of the pool is releasable.
   * 3. Create a Blocks Releasable heuristic function that will coalesce when 3 blocks of the pool are releasable.
   * 4. Create a Blocks Releasable heuristic function that will coalesce when 5 blocks of the pool are releasable.
   */
  auto pr75_hwm_heuristic = umpire::strategy::QuickPool::percent_releasable_hwm(75);
  auto pr100_heuristic = umpire::strategy::QuickPool::percent_releasable(100);
  auto br3_heuristic = umpire::strategy::QuickPool::blocks_releasable(3);
  auto br5_hwm_heuristic = umpire::strategy::QuickPool::blocks_releasable_hwm(5);

  //Note: if no heuristic function is set for a pool, the default heuristic function is Percent Releasable set to 100%.
  //This should work decently well for many cases, but with particular allocation patterns, it may not be aggressive
  //enough.

  /*
   * Below, create a separate QuickPool for each heuristic function. The pools should have a parameter to set the
   * size of the first block in the pool, a parameter to set the size of the next blocks in the pool, the alignment, and
   * finaly the heuristic function.
   *
   * By passing the specific heuristic function to the constructor of the pool, we are ensuring that every time the
   * pool must be coalesced, it uses the exact heuristic function we set above.
   */
  auto pool1 = rm.makeAllocator<umpire::strategy::QuickPool>("pool1", allocator, 1024ul, 1024ul, 16, pr75_hwm_heuristic);
  auto pool2 = rm.makeAllocator<umpire::strategy::QuickPool>("pool2", allocator, 1024ul, 1024ul, 16, pr100_heuristic);
  auto pool3 = rm.makeAllocator<umpire::strategy::QuickPool>("pool3", allocator, 1024ul, 1024ul, 16, br3_heuristic);
  auto pool4 = rm.makeAllocator<umpire::strategy::QuickPool>("pool4", allocator, 1024ul, 1024ul, 16, br5_hwm_heuristic);

  //Note: below we are using the allocator's Unwrap utility to expose the QuickPool class underneath. We will use
  //this to query pool stats below. It is not a requirement, but can be useful for debugging.
  auto quick_pool1 = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool1);
  auto quick_pool2 = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool2);
  auto quick_pool3 = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool3);
  auto quick_pool4 = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool4);

  //Allocate 4 arrays of void pointers
  void *a[4], *b[4], *c[4], *d[4];
  
  //Allocate 1024 bytes in each element of each array
  for (int i = 0; i < 4; ++i) {
    a[i] = pool1.allocate(1024);
    b[i] = pool2.allocate(1024);
    c[i] = pool3.allocate(1024);
    d[i] = pool4.allocate(1024);
  }

  //Only deallocate one element of the array so that one block is freed up.
  pool1.deallocate(a[1]);
  pool2.deallocate(b[1]);
  pool3.deallocate(c[1]);
  pool4.deallocate(d[1]);

  //Allocate larger amounts of bytes in its place. This will cause the pool to rearrange blocks under the hood.
  a[1] = pool1.allocate(4096);
  b[1] = pool2.allocate(4096);
  c[1] = pool3.allocate(4096);
  d[1] = pool4.allocate(4096);

  //Next, deallocate another element of the array that is different from above.
  pool1.deallocate(a[2]);
  pool2.deallocate(b[2]);
  pool3.deallocate(c[2]);
  pool4.deallocate(d[2]);

  //Allocate a smaller amount of bytes in its place. This will cause the pool to rearrange blocks under the hood.
  a[2] = pool1.allocate(64);
  b[2] = pool2.allocate(64);
  c[2] = pool3.allocate(64);
  d[2] = pool4.allocate(64);

  //As we deallocate from each pool, print out stats. Each pool should behave different under the hood because
  //of the different coalescing heuristic functions used.
  for (int i = 0; i < 4; ++i) {
    pool1.deallocate(a[i]);
    std::cout << "Pool1 has " << pool1.getActualSize() << " bytes of memory. "
              << pool1.getCurrentSize() << " bytes are used. " << quick_pool1->getBlocksInPool()
              << " blocks are in the pool. " << quick_pool1->getReleasableSize() << " bytes are releaseable. "
              << std::endl; 
  }
  std::cout << "----------------------------------" << std::endl;

  for (int i = 0; i < 4; ++i) {
    pool2.deallocate(b[i]);
    std::cout << "Pool2 has " << pool2.getActualSize() << " bytes of memory. "
              << pool2.getCurrentSize() << " bytes are used. " << quick_pool2->getBlocksInPool()
              << " blocks are in the pool. " << quick_pool2->getReleasableSize() << " bytes are releaseable. "
              << std::endl;
  }
  std::cout << "----------------------------------" << std::endl;

  for (int i = 0; i < 4; ++i) {
    pool3.deallocate(c[i]);
    std::cout << "Pool3 has " << pool3.getActualSize() << " bytes of memory. "
              << pool3.getCurrentSize() << " bytes are used. " << quick_pool3->getBlocksInPool()
              << " blocks are in the pool. " << quick_pool3->getReleasableSize() << " bytes are releaseable. "
              << std::endl;
  }
  std::cout << "----------------------------------" << std::endl;

  for (int i = 0; i < 4; ++i) {
    pool4.deallocate(d[i]);
    std::cout << "Pool4 has " << pool4.getActualSize() << " bytes of memory. "
              << pool4.getCurrentSize() << " bytes are used. " << quick_pool4->getBlocksInPool()
              << " blocks are in the pool. " << quick_pool4->getReleasableSize() << " bytes are releaseable. "
              << std::endl;
  }
  std::cout << "----------------------------------" << std::endl;

  return 0;
}
