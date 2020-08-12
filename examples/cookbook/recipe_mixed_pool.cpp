//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/MixedPool.hpp"

int main(int, char **)
{
  auto &rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  /*
   * Create a default mixed pool.
   */
  auto default_mixed_allocator = rm.makeAllocator<umpire::strategy::MixedPool>(
      "default_mixed_pool", allocator);

  UMPIRE_USE_VAR(default_mixed_allocator);

  /*
   * Create a mixed pool using fixed pool bins of size 2^8 = 256 Bytes
   * to 2^14 = 16 kB in increments of 5x, where each individual fixed
   * pool is kept under 4MB in size to begin.
   */
  auto custom_mixed_allocator = rm.makeAllocator<umpire::strategy::MixedPool>(
      "custom_mixed_pool", allocator, 256, 16 * 1024, 4 * 1024 * 1024, 5);

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
