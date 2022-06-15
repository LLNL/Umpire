//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/util/wrap_allocator.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  auto heuristic = umpire::strategy::QuickPool::percent_releasable(75);
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("host_quick_pool", allocator, 1024ul, 1024ul, 16, heuristic);

  auto unwrap_quick_pool = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool);


  // Allocate memory in pool
  void* a[4];
  for (int i = 0; i < 4; ++i)
    a[i] = nullptr;

  // Create Fragmentation by deallocating a few allocations and reallocating to different values
  // pool.deallocate(a[1]);
  // pool.deallocate(a[2]);
  a[1] = nullptr;
  a[2] = nullptr;

  // Coalesce the pool
  unwrap_quick_pool->coalesce();

  return (int)(a[1] == a[2]);
}
