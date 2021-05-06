//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/QuickPool.hpp"

namespace {
  template <typename T>
  struct PoolName {
    static constexpr const char* value = "Unknown";
  };

  template <>
  struct PoolName<umpire::strategy::DynamicPoolList> {
    static constexpr const char* value = "DynamicPoolList";
  };

  template <>
  struct PoolName<umpire::strategy::QuickPool> {
    static constexpr const char* value = "QuickPool";
  };

}

template <typename POOL>
struct PoolHeuristicsTest : public ::testing::Test {
  using myPoolType = POOL;
};

using PoolTypes = testing::Types<umpire::strategy::DynamicPoolList, umpire::strategy::QuickPool>;

TYPED_TEST_SUITE(PoolHeuristicsTest, PoolTypes, );

TYPED_TEST(PoolHeuristicsTest, PercentReleasable)
{
  using myPoolType = typename TestFixture::myPoolType;
  const std::string
    pool_name{"Percent_" + std::string{PoolName<myPoolType>::value}};

  auto& rm = umpire::ResourceManager::getInstance();
  auto resource = rm.getAllocator("HOST");
  const std::size_t first_block{1024};
  const std::size_t next_block{128};
  const std::size_t alignment{16};

  using CoalesceHeuristic = std::function<bool(const myPoolType &)>;

  CoalesceHeuristic heuristic{myPoolType::percent_releasable(100)};

  auto pool{
    rm.makeAllocator<myPoolType>
      (pool_name, resource, first_block, next_block, alignment, heuristic)};

  auto strategy = pool.getAllocationStrategy();
  myPoolType* qp_strat{dynamic_cast<myPoolType*>(strategy)};

  ASSERT_NE(qp_strat, nullptr);

  std::vector<void*> ptrs;
  const int max_blocks{9};

  for (int i{0}; i < max_blocks; i++) {
    ASSERT_NO_THROW( ptrs.push_back(pool.allocate(first_block)); );
    ASSERT_EQ(qp_strat->getReleasableBlocks(), 0);
    ASSERT_EQ(qp_strat->getTotalBlocks(), i+1);
  }

  for (int i{max_blocks-1}; i > 0; i--) {
    ASSERT_NO_THROW( pool.deallocate(ptrs[i]); );
    ASSERT_EQ(qp_strat->getReleasableBlocks(), max_blocks-i);
  }

  ASSERT_NO_THROW( pool.deallocate(ptrs[0]); );
  ASSERT_EQ(qp_strat->getReleasableBlocks(), 1);
  ASSERT_EQ(qp_strat->getTotalBlocks(), 1);
}

TYPED_TEST(PoolHeuristicsTest, BlocksReleasable)
{
  using myPoolType = typename TestFixture::myPoolType;
  const std::string
    pool_name{"Blocks_" + std::string{PoolName<myPoolType>::value}};

  auto& rm = umpire::ResourceManager::getInstance();
  auto resource = rm.getAllocator("HOST");
  const std::size_t first_block{1024};
  const std::size_t next_block{128};
  const std::size_t alignment{16};

  using CoalesceHeuristic = std::function<bool(const myPoolType &)>;

  CoalesceHeuristic heuristic{myPoolType::blocks_releasable(2)};

  auto pool{
    rm.makeAllocator<myPoolType>
      (pool_name, resource, first_block, next_block, alignment, heuristic)};

  auto strategy = pool.getAllocationStrategy();
  myPoolType* qp_strat{dynamic_cast<myPoolType*>(strategy)};

  ASSERT_NE(qp_strat, nullptr);

  std::vector<void*> ptrs;
  const int max_blocks{9};

  for (int i{0}; i < max_blocks; i++) {
    ASSERT_NO_THROW( ptrs.push_back(pool.allocate(first_block)); );
    ASSERT_EQ(qp_strat->getReleasableBlocks(), 0);
    ASSERT_EQ(qp_strat->getTotalBlocks(), i+1);
  }

  for (int i{max_blocks}; i > 0; i--) {
    ASSERT_NO_THROW( pool.deallocate(ptrs[i-1]); );

    if (i % 2)
      ASSERT_EQ(qp_strat->getReleasableBlocks(), 1);
    else
      ASSERT_EQ(qp_strat->getReleasableBlocks(), 2);
  }
  ASSERT_EQ(qp_strat->getReleasableBlocks(), 1);
  ASSERT_EQ(qp_strat->getTotalBlocks(), 1);
}

