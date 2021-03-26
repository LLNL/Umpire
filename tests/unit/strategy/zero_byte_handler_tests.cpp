//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include <iostream>
#include <vector>

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

int main(int, char**)
// TEST(ZeroByteHandlerTest, Construction)
{
  std::size_t max_allocations{56};
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");
  std::vector<void*> allocations;

  std::size_t i{0};
  for ( ; i < max_allocations; i++) {
    allocations.push_back( alloc.allocate(0) );
  }

  if (i != max_allocations) {
    std::cout << "Allocation #" << i << " failed" << std::endl;
  }

  if ( rm.hasAllocator(allocations[1]) != true ) {
    std::cout << "OOPS, Allocation " << allocations[i] << " already missing" << std::endl;
    return 1;
  }

  //
  // This should cause a failure
  //
  allocations.push_back( alloc.allocate(0) );

  if ( rm.hasAllocator(allocations[1]) != true ) {
    std::cout << allocations[1] << " Not found" << std::endl;
    return 2;
  }
  else {
    std::cout << allocations[1] << " Hmm, Allocated and Found" << std::endl;
  }

#if 0
  for ( auto& a : allocations ) {
    alloc.deallocate(a);
  }
#endif
}
