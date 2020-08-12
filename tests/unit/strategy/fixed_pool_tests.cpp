//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/FixedPool.hpp"

TEST(FixedPoolTest, Construction)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  umpire::strategy::FixedPool pool{"FixedPool", 0, alloc, sizeof(int)};

  EXPECT_EQ(pool.getCurrentSize(), 0);
}

TEST(FixedPoolTest, Construction_with_count)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  const int num_obj{1124622};

  umpire::strategy::FixedPool pool{"FixedPool", 0, alloc, sizeof(int), num_obj};

  EXPECT_GT(pool.getActualSize(), num_obj * sizeof(int));
}

TEST(FixedPoolTest, Allocate)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  umpire::strategy::FixedPool pool{"FixedPool", 0, alloc, sizeof(int)};

  void* ptr1{pool.allocate(0)};
  void* ptr2{pool.allocate(0)};

  EXPECT_EQ(pool.getCurrentSize(), 2 * sizeof(int));
  EXPECT_EQ(pool.getHighWatermark(), 2 * sizeof(int));

  pool.deallocate(ptr1);

  EXPECT_EQ(pool.getCurrentSize(), 1 * sizeof(int));

  pool.deallocate(ptr2);

  EXPECT_EQ(pool.getCurrentSize(), 0);
  EXPECT_EQ(pool.getHighWatermark(), 2 * sizeof(int));
}

TEST(FixedPoolTest, Allocate_2_pools)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  umpire::strategy::FixedPool pool{"FixedPool", 0, alloc, sizeof(int), 2};

  void* ptr1{pool.allocate()};
  void* ptr2{pool.allocate(sizeof(int))};

  EXPECT_EQ(pool.numPools(), 1);

  void* ptr3{pool.allocate()};

  EXPECT_EQ(pool.numPools(), 2);
  EXPECT_EQ(pool.getCurrentSize(), 3 * sizeof(int));
  EXPECT_EQ(pool.getHighWatermark(), 3 * sizeof(int));

  pool.deallocate(ptr1);

  EXPECT_EQ(pool.getCurrentSize(), 2 * sizeof(int));

  pool.deallocate(ptr2);
  pool.deallocate(ptr3);

  EXPECT_EQ(pool.getCurrentSize(), 0);
  EXPECT_EQ(pool.getHighWatermark(), 3 * sizeof(int));
}
