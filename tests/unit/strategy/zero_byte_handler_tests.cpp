//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <vector>

#include "gtest/gtest.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

TEST(ZeroByteHandlerTest, Construction)
{
  std::size_t max_allocations{1000};
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");
  std::vector<void*> allocations;

  std::size_t i{0};
  for (; i < max_allocations; i++) {
    ASSERT_NO_THROW(allocations.push_back(alloc.allocate(0)));
  }

  for (auto& a : allocations) {
    ASSERT_TRUE(rm.hasAllocator(a) == true);
    ASSERT_NO_THROW(alloc.deallocate(a));
  }
}
