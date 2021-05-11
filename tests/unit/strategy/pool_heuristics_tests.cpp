//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include <sstream>
#include <string>
#include <utility>

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
  const std::size_t first_block{1024};
  const std::size_t next_block{128};
  const std::size_t alignment{16};

  using myPoolType = POOL;
  using CoalesceHeuristic = std::function<bool(const POOL &)>;
  using TestAllocator = std::pair<umpire::Allocator, POOL*>;

  TestAllocator getAllocator(CoalesceHeuristic h)
  {
    auto& rm = umpire::ResourceManager::getInstance();
    TestAllocator rval;
    static int unique_id{0};
    std::stringstream n;

    n << "Pool_" << std::string{PoolName<myPoolType>::value} << "_" << unique_id++;

    rval.first = rm.makeAllocator<myPoolType>(n.str(), rm.getAllocator("HOST"), first_block, next_block, alignment, h);
    auto strategy = rval.first.getAllocationStrategy();
    rval.second = dynamic_cast<POOL*>(strategy);
    return rval;
  }
};

using PoolTypes = testing::Types<umpire::strategy::DynamicPoolList, umpire::strategy::QuickPool>;

TYPED_TEST_SUITE(PoolHeuristicsTest, PoolTypes, );

TYPED_TEST(PoolHeuristicsTest, PercentReleasable)
{
  using myPoolType = typename TestFixture::myPoolType;
  using TestAllocator = typename TestFixture::TestAllocator;
  TestAllocator a;

  ASSERT_NO_THROW( a = this->getAllocator(myPoolType::percent_releasable(100)); );
  ASSERT_NE(a.second, nullptr);

  std::vector<void*> ptrs;
  const int max_blocks{9};

  for (int i{0}; i < max_blocks; i++) {
    ASSERT_NO_THROW( ptrs.push_back(a.first.allocate(this->first_block)); );
    ASSERT_EQ(a.second->getReleasableBlocks(), 0);
    ASSERT_EQ(a.second->getTotalBlocks(), i+1);
  }

  for (int i{max_blocks-1}; i > 0; i--) {
    ASSERT_NO_THROW( a.first.deallocate(ptrs[i]); );
    ASSERT_EQ(a.second->getReleasableBlocks(), max_blocks-i);
  }

  ASSERT_NO_THROW( a.first.deallocate(ptrs[0]); );
  ASSERT_EQ(a.second->getReleasableBlocks(), 1);
  ASSERT_EQ(a.second->getTotalBlocks(), 1);
}

TYPED_TEST(PoolHeuristicsTest, BlocksReleasable)
{
  using myPoolType = typename TestFixture::myPoolType;
  using TestAllocator = typename TestFixture::TestAllocator;
  TestAllocator a;

  ASSERT_NO_THROW( a = this->getAllocator(myPoolType::blocks_releasable(2)); );
  ASSERT_NE(a.second, nullptr);

  std::vector<void*> ptrs;
  const int max_blocks{9};

  for (int i{0}; i < max_blocks; i++) {
    ASSERT_NO_THROW( ptrs.push_back(a.first.allocate(this->first_block)); );
    ASSERT_EQ(a.second->getReleasableBlocks(), 0);
    ASSERT_EQ(a.second->getTotalBlocks(), i+1);
  }

  for (int i{max_blocks}; i > 0; i--) {
    ASSERT_NO_THROW( a.first.deallocate(ptrs[i-1]); );

    if (i % 2)
      ASSERT_EQ(a.second->getReleasableBlocks(), 1);
    else
      ASSERT_EQ(a.second->getReleasableBlocks(), 2);
  }
  ASSERT_EQ(a.second->getReleasableBlocks(), 1);
  ASSERT_EQ(a.second->getTotalBlocks(), 1);
}
