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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "umpire/config.hpp"

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/Allocator.hpp"

class MockAllocationStrategy : public umpire::strategy::AllocationStrategy
{
  public:
    MockAllocationStrategy() :
      umpire::strategy::AllocationStrategy("MockAllocationStrategy", 12345)
    {
    }

    MOCK_METHOD1(allocate, void*(std::size_t bytes));
    MOCK_METHOD1(deallocate, void(void* ptr));
    MOCK_METHOD0(getCurrentSize, long() const noexcept);
    MOCK_METHOD0(getHighWatermark, long() const noexcept);
    MOCK_METHOD0(getPlatform, umpire::Platform() noexcept);
};

class AllocatorTest : public ::testing::Test
{
protected:
  AllocatorTest():
    m_strategy(std::make_shared<MockAllocationStrategy>()),
    m_allocator(m_strategy)
  {
  }

  virtual void SetUp()
  {
    data = malloc(100*sizeof(char));

    ON_CALL(*m_strategy, getPlatform())
      .WillByDefault(::testing::Return(umpire::Platform::cpu));

    // set up allocate return value
    ON_CALL(*m_strategy, allocate(::testing::_))
      .WillByDefault(::testing::Return(data));
  }

  virtual void TearDown()
  {
    free(data);
  }

  strategy::AllocationStrategy* m_strategy;
  umpire::Allocator m_allocator;
  void* data;
};

TEST_F(AllocatorTest, getName)
{
  ASSERT_EQ(m_allocator.getName(), "MockAllocationStrategy");
}

TEST_F(AllocatorTest, getId)
{
  ASSERT_EQ(m_allocator.getId(), 12345);
}

TEST_F(AllocatorTest, getPlatform)
{
  EXPECT_CALL(*m_strategy, getPlatform());

  ASSERT_EQ(m_allocator.getPlatform(), umpire::Platform::cpu);
}

TEST_F(AllocatorTest, allocate)
{
  EXPECT_CALL(*m_strategy, allocate(64));

  char* my_data = static_cast<char*>(m_allocator.allocate(64));

  ASSERT_NE(my_data, nullptr);
}

TEST_F(AllocatorTest, deallocate)
{
  EXPECT_CALL(*m_strategy, allocate(64));

  char* my_data = static_cast<char*>(m_allocator.allocate(64));

  EXPECT_CALL(*m_strategy, deallocate(my_data));

  m_allocator.deallocate(my_data);
}

TEST_F(AllocatorTest, getAllocationStrategy)
{
  ASSERT_EQ(m_strategy, m_allocator.getAllocationStrategy());
}

TEST_F(AllocatorTest, getCurrentSize)
{
  EXPECT_CALL(*m_strategy, getCurrentSize())
    .WillOnce(::testing::Return(1024));

  int size = m_allocator.getCurrentSize();

  ASSERT_EQ(size, 1024);
}

TEST_F(AllocatorTest, getHighWatermark)
{
  EXPECT_CALL(*m_strategy, getHighWatermark())
    .WillOnce(::testing::Return(4096));

  int size = m_allocator.getHighWatermark();

  ASSERT_EQ(size, 4096);
}
