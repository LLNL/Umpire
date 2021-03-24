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
  std::size_t max_allocations{1024};
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");
  std::vector<void*> allocations;

  for (std::size_t i{0}; i < max_allocations; i++) {
    allocations.push_back( alloc.allocate(0) );
    for (std::size_t j{0}; j <= i; j++) {
      if ( rm.hasAllocator(allocations[j]) != true ) {
        std::cout << allocations[j] << " Not found" << std::endl;
        return 1;
      }
      else {
        if ( j == i ) {
          std::cout << allocations[j] << " Allocated and Found" << std::endl;
        }
      }
    }
  }
  std::cout << "Done!" << std::endl;

#if 0
  for ( auto& a : allocations ) {
    alloc.deallocate(a);
  }
#endif
}
