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
#include "umpire/util/MemoryMap.hpp"

#include "gtest/gtest.h"

using Map = umpire::util::MemoryMap<int>;

static size_t size_by_iterator(Map& map)
{
  auto iter = map.begin(), end = map.end();
  size_t size = 0;
  while (iter != end) { ++size; ++iter; }
  return size;
}

TEST(MemoryMap, get)
{
  Map map{};

  void* a = reinterpret_cast<void*>(1);

  // Get should emplace an entry if it doens't already exist
  Map::Iterator iter{map.end()};
  bool found;

  {
    std::tie(iter, found) = map.get(a, 1);
    EXPECT_EQ(iter, map.begin());
    EXPECT_EQ(found, false);
  }

  {
    std::tie(iter, found) = map.get(a, 1);
    EXPECT_EQ(iter, map.begin());
    EXPECT_EQ(found, true);
  }

  ASSERT_EQ(map.size(), 1);
}

TEST(MemoryMap, insert)
{
  Map map{};

  void* a = reinterpret_cast<void*>(1);
  void* b = reinterpret_cast<void*>(2);
  EXPECT_NO_THROW(
    map.insert(a, 1);
    map.insert(b, 2));

  EXPECT_THROW(map.insert(b, 2), umpire::util::Exception);

  ASSERT_EQ(map.size(), 2);
  ASSERT_EQ(size_by_iterator(map), 2);
}

TEST(MemoryMap, find)
{
  Map map{};

  void* a = reinterpret_cast<void*>(1);
  void* b = reinterpret_cast<void*>(2);
  void* c = reinterpret_cast<void*>(3);
  map.insert(a, 1);
  map.insert(b, 2);

  {
    auto iter = map.find(a);
    EXPECT_EQ(iter->first, a);
  }

  {
    auto iter = map.find(c);
    EXPECT_EQ(iter, map.end());
    EXPECT_EQ(iter->second, nullptr);
  }
}

TEST(MemoryMap, findOrBefore)
{
  Map map{};

  void* a = reinterpret_cast<void*>(1);
  map.insert(a, 1);

  {
    auto iter = map.findOrBefore(static_cast<char*>(a) + 10);
    EXPECT_EQ(iter->first, a);
  }

  map.remove(a);
  {
    auto iter = map.findOrBefore(static_cast<char*>(a) + 10);
    EXPECT_EQ(iter, map.begin());
    EXPECT_EQ(iter, map.end());
  }

  ASSERT_EQ(map.size(), 0);
}

TEST(MemoryMap, remove)
{
  Map map{};

  void* a = reinterpret_cast<void*>(1);
  map.insert(a, 1);

  ASSERT_NO_THROW(map.remove(a));
  ASSERT_ANY_THROW(map.remove(a));
  ASSERT_EQ(map.size(), 0);
}

TEST(MemoryMap, clear)
{
  Map map{};

  void* a = reinterpret_cast<void*>(1);
  void* b = reinterpret_cast<void*>(2);
  void* c = reinterpret_cast<void*>(3);
  map.insert(a, 1);
  map.insert(b, 2);
  map.insert(c, 3);

  ASSERT_NO_THROW(map.remove(a));

  map.clear();
  ASSERT_EQ(map.size(), 0);
  ASSERT_EQ(size_by_iterator(map), 0);
}


TEST(MemoryMap, Iterator)
{
  Map map{};

  void* a = reinterpret_cast<void*>(1);
  auto ia = map.insert(a, 1);

  auto begin = map.begin();
  ASSERT_EQ(begin->first, a);
  ASSERT_EQ(ia, map.begin());
  ASSERT_EQ(++ia, map.end());
}

TEST(MemoryMap, ordering)
{
  Map map{};

  void* a = reinterpret_cast<void*>(1);
  void* b = reinterpret_cast<void*>(2);
  auto ia = map.insert(a, 1);
  auto ib = map.insert(b, 2);
  ASSERT_EQ(++ia, ib);
}
