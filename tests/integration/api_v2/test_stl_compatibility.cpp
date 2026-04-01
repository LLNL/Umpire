//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/allocator.hpp"
#include "umpire/resource/host_memory.hpp"

#include <gtest/gtest.h>

#include <deque>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using host_memory = umpire::resource::host_memory<>;

template<typename T>
using host_allocator = umpire::allocator<T, host_memory>;

host_memory& host()
{
  return host_memory::get();
}

} // namespace

TEST(STLCompatibility, VectorReserveResizeAndPushBack)
{
  std::vector<int, host_allocator<int>> values{host_allocator<int>{&host()}};

  values.reserve(16);
  EXPECT_GE(values.capacity(), 16u);

  values.resize(4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int>(i + 1);
  }

  values.push_back(9);

  ASSERT_EQ(values.size(), 5u);
  EXPECT_EQ(values.front(), 1);
  EXPECT_EQ(values.back(), 9);
  EXPECT_EQ(values.get_allocator().get_memory(), &host());
}

TEST(STLCompatibility, MapStoresOrderedKeyValuePairs)
{
  using value_type = std::pair<const int, std::string>;
  using allocator_type = host_allocator<value_type>;

  std::map<int, std::string, std::less<int>, allocator_type> values{allocator_type{&host()}};
  values.emplace(4, "four");
  values.emplace(2, "two");
  values.emplace(3, "three");

  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values.begin()->first, 2);
  EXPECT_EQ(values.at(3), "three");
}

TEST(STLCompatibility, AllocatorExtendedConstructorsInitializeContainers)
{
  std::vector<int, host_allocator<int>> values(3, 7, host_allocator<int>{&host()});

  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0], 7);
  EXPECT_EQ(values[2], 7);
  EXPECT_EQ(values.get_allocator().get_memory(), &host());
}

TEST(STLCompatibility, UnorderedMapStoresAndFindsValues)
{
  using value_type = std::pair<const int, std::string>;
  using allocator_type = host_allocator<value_type>;

  std::unordered_map<int, std::string, std::hash<int>, std::equal_to<int>, allocator_type> values{
    0, std::hash<int>{}, std::equal_to<int>{}, allocator_type{&host()}};

  values.emplace(7, "seven");
  values.emplace(11, "eleven");

  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values.at(7), "seven");
  EXPECT_NE(values.find(11), values.end());
}

TEST(STLCompatibility, ListSupportsPushFrontAndPushBack)
{
  std::list<int, host_allocator<int>> values{host_allocator<int>{&host()}};

  values.push_back(2);
  values.push_front(1);
  values.push_back(3);

  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values.front(), 1);
  EXPECT_EQ(values.back(), 3);
}

TEST(STLCompatibility, DequeSupportsMixedInsertionAndIndexing)
{
  std::deque<int, host_allocator<int>> values{host_allocator<int>{&host()}};

  values.push_back(2);
  values.push_front(1);
  values.push_back(3);

  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], 2);
  EXPECT_EQ(values[2], 3);
}

TEST(STLCompatibility, AllocateSharedUsesAllocatorRebind)
{
  auto value = std::allocate_shared<std::string>(host_allocator<std::string>{&host()}, "api_v2");

  ASSERT_TRUE(static_cast<bool>(value));
  EXPECT_EQ(*value, "api_v2");
}

TEST(STLCompatibility, MoveConstructionPreservesAllocatorAndContents)
{
  std::vector<int, host_allocator<int>> source{host_allocator<int>{&host()}};
  source.push_back(5);
  source.push_back(8);

  std::vector<int, host_allocator<int>> moved{std::move(source)};

  ASSERT_EQ(moved.size(), 2u);
  EXPECT_EQ(moved[0], 5);
  EXPECT_EQ(moved[1], 8);
  EXPECT_EQ(moved.get_allocator().get_memory(), &host());
}

TEST(STLCompatibility, MoveAssignmentPreservesContentsWithSharedAllocator)
{
  using vector_type = std::vector<int, host_allocator<int>>;

  vector_type source{host_allocator<int>{&host()}};
  vector_type destination{host_allocator<int>{&host()}};

  source.push_back(4);
  source.push_back(6);
  destination.push_back(1);

  destination = std::move(source);

  ASSERT_EQ(destination.size(), 2u);
  EXPECT_EQ(destination[0], 4);
  EXPECT_EQ(destination[1], 6);
  EXPECT_EQ(destination.get_allocator().get_memory(), &host());
}

TEST(STLCompatibility, SwapExchangesContentsWithSharedAllocator)
{
  using vector_type = std::vector<int, host_allocator<int>>;

  vector_type left{host_allocator<int>{&host()}};
  vector_type right{host_allocator<int>{&host()}};

  left.push_back(1);
  left.push_back(2);
  right.push_back(9);

  left.swap(right);

  ASSERT_EQ(left.size(), 1u);
  ASSERT_EQ(right.size(), 2u);
  EXPECT_EQ(left.front(), 9);
  EXPECT_EQ(right.front(), 1);
  EXPECT_EQ(left.get_allocator().get_memory(), &host());
  EXPECT_EQ(right.get_allocator().get_memory(), &host());
}
