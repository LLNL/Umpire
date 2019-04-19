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
#include "umpire/strategy/MixedPool.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include <iostream>

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  /*
   * Create a default mixed pool.
   */
  auto default_mixed_allocator = rm.makeAllocator<umpire::strategy::MixedPool>(
    "default_mixed_pool", allocator);

  UMPIRE_USE_VAR(default_mixed_allocator);

  /*
   * Create a mixed pool using fixed pool bins of size 2^8 = 256 Bytes
   * to 2^14 = 16 kB in increments of powers of 2, skipping every other power.
   */
  auto custom_mixed_allocator = rm.makeAllocator<umpire::strategy::MixedPoolImpl<8,2,14>>(
    "custom_mixed_pool", allocator);

  /*
   * Although this calls for only 4*4=16 bytes, this allocation will
   * come from the smallest fixed pool, thus ptr will actually be the
   * first address in a range of 256 bytes.
   */
  void *ptr1 = custom_mixed_allocator.allocate(4 * sizeof(int));

  /*
   * This is too beyond the range of the fixed pools, and therefore is
   * allocated from a dynamic pool. The range of address space
   * reserved will be exactly what was requested by the allocate()
   * method.
   */
  void *ptr2 = custom_mixed_allocator.allocate(1 << 18);

  /*
   * Clean up
   */
  custom_mixed_allocator.deallocate(ptr1);
  custom_mixed_allocator.deallocate(ptr2);

  return 0;
}
