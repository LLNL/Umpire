//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/util/Macros.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  //
  // Create a pool with introspection disabled (can improve performance)
  //
  // _umpire_tut_nointro_start
  auto pooled_allocator =
      rm.makeAllocator<umpire::strategy::DynamicPool, false>(
          "NO_INTROSPECTION_POOL", allocator);
  // _umpire_tut_nointro_end

  void* data = pooled_allocator.allocate(1024);

  // _umpire_tut_getsize_start
  std::cout << "Pool has allocated " << pooled_allocator.getActualSize()
            << " bytes of memory. " << pooled_allocator.getCurrentSize()
            << " bytes are used" << std::endl;
  // _umpire_tut_getsize_end

  pooled_allocator.deallocate(data);

  return 0;
}
