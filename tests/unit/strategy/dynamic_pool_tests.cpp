//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/DynamicPool.hpp"

static constexpr std::size_t SIZE = 1024;

TEST(DynamicPoolTest, Construction) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  {
    umpire::strategy::DynamicPool pool{"DynamicPool", 0, alloc};
    // Pool should pre-allocate some memory by default
    EXPECT_GT(pool.getActualSize(), 0);

    // But there should be no live allocations
    EXPECT_EQ(pool.getCurrentSize(), 0);
  }

  {
    umpire::strategy::DynamicPool pool{"DynamicPool", 0, alloc,
                                       SIZE*SIZE, SIZE, umpire::strategy::heuristic_percent_releasable(100), 16};

    // Pool should pre-allocate exactly this amount of memory (assuming alignment fits)
    EXPECT_EQ(pool.getActualSize(), SIZE*SIZE);

    // But there should be no live allocations
    EXPECT_EQ(pool.getCurrentSize(), 0);
  }
}

TEST(DynamicPoolTest, Allocate) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  {
    umpire::strategy::DynamicPool pool{"DynamicPool", 0, alloc};

    void* ptr1{pool.allocate(SIZE)};
    EXPECT_EQ(pool.getCurrentSize(), 1*SIZE);

    void* ptr2{pool.allocate(SIZE)};
    EXPECT_EQ(pool.getCurrentSize(), 2*SIZE);

    // Pool internal size should be greater than or equal to the current size
    EXPECT_GE(pool.getActualSize(), 2*SIZE);

    pool.deallocate(ptr1);
    pool.deallocate(ptr2);

    EXPECT_EQ(pool.getCurrentSize(), 0);

    // Pool should hang on to the memory
    EXPECT_GE(pool.getActualSize(), 2*SIZE);
  }

  // Allocate with a min_alloc_size greater than the size requested
  {
    umpire::strategy::DynamicPool pool{"DynamicPool", 0, alloc, 64, SIZE*SIZE};

    // Should allocate a SIZE*SIZE block, leaving the initial block in the pool
    void* ptr{pool.allocate(SIZE)};

    EXPECT_EQ(pool.getCurrentSize(), 1*SIZE);
    EXPECT_EQ(pool.getActualSize(), 64 + SIZE*SIZE);

    pool.deallocate(ptr);
  }
}

TEST(DynamicPoolTest, release) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  {
    umpire::strategy::DynamicPool pool{"DynamicPool", 0, alloc};

    void* ptr{pool.allocate(SIZE)};

    pool.release();

    EXPECT_EQ(pool.getCurrentSize(), SIZE);
    EXPECT_GE(pool.getActualSize(), SIZE);

    pool.deallocate(ptr);
    pool.release();

    EXPECT_EQ(pool.getCurrentSize(), 0);
    EXPECT_EQ(pool.getActualSize(), 0);
  }
}
