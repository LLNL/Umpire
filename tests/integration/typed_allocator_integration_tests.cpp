//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/TypedAllocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/MemoryResourceTypes.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/SizeLimiter.hpp"

class TypedAllocatorTest : public ::testing::TestWithParam<std::string> {
 public:
  virtual void SetUp()
  {
    auto& rm = umpire::ResourceManager::getInstance();
    m_allocator = new umpire::TypedAllocator<double>(
        rm.getAllocator(GetParam()));
  }

  virtual void TearDown()
  {
    delete m_allocator;
  }

  umpire::TypedAllocator<double>* m_allocator;

  const std::size_t m_big = 64;
  const std::size_t m_small = 8;
  const std::size_t m_nothing = 0;
};

TEST_P(TypedAllocatorTest, AllocateDeallocateBig)
{
  double* data = m_allocator->allocate(m_big);

  ASSERT_NE(nullptr, data);

  m_allocator->deallocate(data, m_big);
}

TEST_P(TypedAllocatorTest, AllocateDeallocateSmall)
{
  double* data = m_allocator->allocate(m_small);

  ASSERT_NE(nullptr, data);

  m_allocator->deallocate(data, m_small);
}

TEST_P(TypedAllocatorTest, AllocateDeallocateNothing)
{
  double* data = m_allocator->allocate(m_nothing);

  ASSERT_NE(nullptr, data);

  m_allocator->deallocate(data, m_nothing);
}

TEST_P(TypedAllocatorTest, DeallocateNullptr)
{
  double* data = nullptr;

  ASSERT_NO_THROW(m_allocator->deallocate(data, 0));

  SUCCEED();
}

TEST_P(TypedAllocatorTest, Equality)
{
  ASSERT_TRUE(((*m_allocator) == (*m_allocator)));

  ASSERT_FALSE(((*m_allocator) != (*m_allocator)));
}

TEST_P(TypedAllocatorTest, EqualityTypeDifferent)
{
  umpire::TypedAllocator<char> alloc_char = *m_allocator;

  ASSERT_TRUE((alloc_char == (*m_allocator)));

  ASSERT_TRUE(((*m_allocator) == alloc_char));

  ASSERT_FALSE((alloc_char != (*m_allocator)));

  ASSERT_FALSE(((*m_allocator) != alloc_char));
}

std::vector<std::string> allocator_strings()
{
  std::vector<std::string> allocators;
  allocators.push_back("HOST");
#if defined(UMPIRE_ENABLE_DEVICE)
  allocators.push_back("DEVICE");
  auto& rm = umpire::ResourceManager::getInstance();
  for (int id = 0; id < rm.getNumDevices(); id++) {
    allocators.push_back(std::string{"DEVICE::" + std::to_string(id)});
  }
#endif
#if defined(UMPIRE_ENABLE_UM)
  allocators.push_back("UM");
#endif
#if defined(UMPIRE_ENABLE_CONST)
  allocators.push_back("DEVICE_CONST");
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  allocators.push_back("PINNED");
#endif

  return allocators;
}

INSTANTIATE_TEST_SUITE_P(Allocators, TypedAllocatorTest,
                         ::testing::ValuesIn(allocator_strings()));


TEST(TypedAllocation, DeallocateDifferent)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::TypedAllocator<double> alloc_one(rm.getAllocator("HOST"));
  umpire::TypedAllocator<double> alloc_two(rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "DeallocateDifferentLimiter", rm.getAllocator("HOST"), 1024));

  double* data = alloc_one.allocate(1024);

  ASSERT_THROW(alloc_two.deallocate(data, 1024), umpire::util::Exception);

  ASSERT_NO_THROW(alloc_one.deallocate(data, 1024));
}


TEST(TypedAllocation, DeallocateDifferentInstance)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::TypedAllocator<double> alloc_one(rm.getAllocator("HOST"));
  umpire::TypedAllocator<double> alloc_two(alloc_one);

  double* data = alloc_one.allocate(1024);

  ASSERT_NO_THROW(alloc_two.deallocate(data, 1024));
}

TEST(TypedAllocation, Equality)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::TypedAllocator<double> alloc_one(rm.getAllocator("HOST"));
  umpire::TypedAllocator<double> alloc_two(rm.getAllocator("HOST"));

  ASSERT_TRUE((alloc_one == alloc_two));

  ASSERT_TRUE((alloc_two == alloc_one));

  ASSERT_FALSE((alloc_one != alloc_two));

  ASSERT_FALSE((alloc_two != alloc_one));
}

TEST(TypedAllocation, EqualityTypeDifferent)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::TypedAllocator<char>   alloc_one(rm.getAllocator("HOST"));
  umpire::TypedAllocator<double> alloc_two(rm.getAllocator("HOST"));

  ASSERT_TRUE((alloc_one == alloc_two));

  ASSERT_TRUE((alloc_two == alloc_one));

  ASSERT_FALSE((alloc_one != alloc_two));

  ASSERT_FALSE((alloc_two != alloc_one));
}

TEST(TypedAllocation, EqualityDifferent)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::TypedAllocator<double> alloc_one(rm.getAllocator("HOST"));
  umpire::TypedAllocator<double> alloc_two(
      rm.makeAllocator<umpire::strategy::SizeLimiter>(
        "EqualityDifferentLimiter", rm.getAllocator("HOST"), 1024));

  ASSERT_FALSE((alloc_one == alloc_two));

  ASSERT_FALSE((alloc_two == alloc_one));

  ASSERT_TRUE((alloc_one != alloc_two));

  ASSERT_TRUE((alloc_two != alloc_one));
}

TEST(TypedAllocation, EqualityDifferentTypeDifferent)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::TypedAllocator<char>   alloc_one(rm.getAllocator("HOST"));
  umpire::TypedAllocator<double> alloc_two(
      rm.makeAllocator<umpire::strategy::SizeLimiter>(
        "EqualityDifferentTypeDifferentLimiter", rm.getAllocator("HOST"), 1024));

  ASSERT_FALSE((alloc_one == alloc_two));

  ASSERT_FALSE((alloc_two == alloc_one));

  ASSERT_TRUE((alloc_one != alloc_two));

  ASSERT_TRUE((alloc_two != alloc_one));
}
