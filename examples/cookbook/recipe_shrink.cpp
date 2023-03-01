//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/util/Macros.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("DEVICE");

  //
  // Create a 4 Gb pool and reserve one word (to maintain aligment)
  //
  // _sphinx_tag_tut_create_pool_start
  auto pooled_allocator =
      rm.makeAllocator<umpire::strategy::QuickPool>("GPU_POOL", allocator, 4ul * 1024ul * 1024ul * 1024ul + 1);
  // _sphinx_tag_tut_create_pool_end

  void* hold = pooled_allocator.allocate(64);
  UMPIRE_USE_VAR(hold);

  std::cout << "Pool has allocated " << pooled_allocator.getActualSize() << " bytes of memory. "
            << pooled_allocator.getCurrentSize() << " bytes are used" << std::endl;

  //
  // Grow pool to ~12 by grabbing a 8Gb chunk
  //
  // _sphinx_tag_tut_grow_pool_start
  void* grow = pooled_allocator.allocate(8ul * 1024ul * 1024ul * 1024ul);
  pooled_allocator.deallocate(grow);

  std::cout << "Pool has allocated " << pooled_allocator.getActualSize() << " bytes of memory. "
            << pooled_allocator.getCurrentSize() << " bytes are used" << std::endl;
  // _sphinx_tag_tut_grow_pool_end

  //
  // Shrink pool back to ~4Gb
  //
  // _sphinx_tag_tut_shrink_pool_back_start
  pooled_allocator.release();
  std::cout << "Pool has allocated " << pooled_allocator.getActualSize() << " bytes of memory. "
            << pooled_allocator.getCurrentSize() << " bytes are used" << std::endl;
  // _sphinx_tag_tut_shrink_pool_back_end

  return 0;
}
