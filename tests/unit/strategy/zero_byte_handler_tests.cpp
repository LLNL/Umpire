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

TEST(ZeroByteHandlerTest, Construction)
{
  std::size_t max_allocations{50000};
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");
  std::vector<void*> allocations;

  for (std::size_t i{0}; i < max_allocations; i++) {
    allocations.push_back( alloc.allocate(0) );
    EXPECT_EQ(true, rm.hasAllocator(allocations[i]));
  }

  for ( auto& a : allocations ) {
    alloc.deallocate(a);
  }
}
