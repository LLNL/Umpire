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
#include "umpire/strategy/DynamicPool.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

#include <iostream>

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("DEVICE");

  /* 
   * Create a 4 Gb pool and reserve one word (to maintain aligment)
   */
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
                            "GPU_POOL",
                            allocator,
                            4ul * 1024ul * 1024ul * 1024ul + 1); 
  void* hold = pooled_allocator.allocate(64);
  UMPIRE_USE_VAR(hold);


  std::cout << "Pool has allocated " << pooled_allocator.getActualSize()
            << " bytes of memory. " << pooled_allocator.getCurrentSize()
            << " bytes are used" << std::endl;


  /*
   * Grow pool to ~12 by grabbing a 8Gb chunk
   */
  void* grow = pooled_allocator.allocate( 8ul * 1024ul * 1024ul * 1024ul );
  pooled_allocator.deallocate(grow);

  std::cout << "Pool has allocated " << pooled_allocator.getActualSize()
            << " bytes of memory. " << pooled_allocator.getCurrentSize()
            << " bytes are used" << std::endl;

  /*
   * Shrink pool back to ~4Gb
   */
  pooled_allocator.release();
  std::cout << "Pool has allocated " << pooled_allocator.getActualSize() 
            << " bytes of memory. " << pooled_allocator.getCurrentSize()
            << " bytes are used" << std::endl;

  return 0;
}
