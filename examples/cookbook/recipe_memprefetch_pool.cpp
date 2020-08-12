//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AllocationPrefetcher.hpp"
#include "umpire/strategy/DynamicPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto prefetcher = rm.makeAllocator<umpire::strategy::AllocationPrefetcher>(
      "prefetcher", rm.getAllocator("UM"));

  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>("prefetch_pool",
                                                              prefetcher);

  void* data_one = pool.allocate(1024);
  void* data_two = pool.allocate(4096);
  void* data_three = pool.allocate(1040);

  pool.deallocate(data_one);
  pool.deallocate(data_two);
  pool.deallocate(data_three);

  return 0;
}
