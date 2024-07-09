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
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("DEVICE");

  //Talk about default percentage. What happens when you set 75% or 50%, etc.
  //Also show percent_releasable_hwm and what that does
  //Also show blocks_releasable and blocks_releasable_hwm
  auto pr75_hwm_heuristic = umpire::strategy::QuickPool::percent_releasable_hwm(75);
  auto pr100_heuristic = umpire::strategy::QuickPool::percent_releasable(100);
  auto br3_heuristic = umpire::strategy::QuickPool::blocks_releasable(3);
  auto br2_hwm_heuristic = umpire::strategy::QuickPool::blocks_releasable_hwm(2);

  auto pool1 = rm.makeAllocator<umpire::strategy::QuickPool>("pool1", allocator, 1024ul, 1024ul, 16, pr75_hwm_heuristic);
  auto pool2 = rm.makeAllocator<umpire::strategy::QuickPool>("pool2", allocator, 1024ul, 1024ul, 16, pr100_heuristic);
  auto pool3 = rm.makeAllocator<umpire::strategy::QuickPool>("pool3", allocator, 1024ul, 1024ul, 16, br3_heuristic);
  auto pool4 = rm.makeAllocator<umpire::strategy::QuickPool>("pool4", allocator, 1024ul, 1024ul, 16, br2_hwm_heuristic);

  //Discuss "unwrapping" the pool allocator
  auto quick_pool1 = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool1);
  auto quick_pool2 = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool2);
  auto quick_pool3 = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool3);
  auto quick_pool4 = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool4);

  //Allocate 4 arrays of void pointers
  void *a[4], *b[4], *c[4], *d[4];
  
  //allocate 1024 bytes in each element of each array
  for (int i = 0; i < 4; ++i) {
    a[i] = pool1.allocate(1024);
    b[i] = pool2.allocate(1024);
    c[i] = pool3.allocate(1024);
    d[i] = pool4.allocate(1024);
  }

  //do computation

  //Only deallocate certain elements of the array but not all
  pool1.deallocate(a[1]);
  pool2.deallocate(b[1]);
  pool3.deallocate(c[1]);
  pool4.deallocate(d[1]);

  //Allocate larger amounts of bytes in its place
  a[1] = pool1.allocate(4096);
  b[1] = pool2.allocate(4096);
  c[1] = pool3.allocate(4096);
  d[1] = pool4.allocate(4096);

  pool1.deallocate(a[2]);
  pool2.deallocate(b[2]);
  pool3.deallocate(c[2]);
  pool4.deallocate(d[2]);

  a[2] = pool1.allocate(64);
  b[2] = pool2.allocate(64);
  c[2] = pool3.allocate(64);
  d[2] = pool4.allocate(64);

  //As we deallocate from each pool, print out stats
  for (int i = 0; i < 4; ++i) {
    pool1.deallocate(a[i]);
    std::cout << "Pool1 has " << pool1.getActualSize() << " bytes of memory. "
              << pool1.getCurrentSize() << " bytes are used. " << quick_pool1->getBlocksInPool()
              << " blocks are in the pool. " << quick_pool1->getReleasableSize() << " bytes are releaseable. "
              << std::endl << "----------------------------------" << std::endl;
  }

  for (int i = 0; i < 4; ++i) {
    pool2.deallocate(b[i]);
    std::cout << "Pool2 has " << pool2.getActualSize() << " bytes of memory. "
              << pool2.getCurrentSize() << " bytes are used. " << quick_pool2->getBlocksInPool()
              << " blocks are in the pool. " << quick_pool2->getReleasableSize() << " bytes are releaseable. "
              << std::endl << "----------------------------------" << std::endl;
  }

  for (int i = 0; i < 4; ++i) {
    pool3.deallocate(c[i]);
    std::cout << "Pool3 has " << pool3.getActualSize() << " bytes of memory. "
              << pool3.getCurrentSize() << " bytes are used. " << quick_pool3->getBlocksInPool()
              << " blocks are in the pool. " << quick_pool3->getReleasableSize() << " bytes are releaseable. "
              << std::endl << "----------------------------------" << std::endl;
  }

  for (int i = 0; i < 4; ++i) {
    pool4.deallocate(d[i]);
    std::cout << "Pool4 has " << pool4.getActualSize() << " bytes of memory. "
              << pool4.getCurrentSize() << " bytes are used. " << quick_pool4->getBlocksInPool()
              << " blocks are in the pool. " << quick_pool4->getReleasableSize() << " bytes are releaseable. "
              << std::endl << "----------------------------------" << std::endl;
  }
  return 0;
}
