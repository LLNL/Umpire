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

TEST(MemoryMap, get)
{
  umpire::util::MemoryMap<int> map{};

  void* a{reinterpret_cast<void*>(1)};

  // Get should emplace an entry if it doens't already exist
  umpire::util::MemoryMap<int>::Iterator<false> iter{&map};
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
  umpire::util::MemoryMap<int> map{};

  void* a{reinterpret_cast<void*>(1)};
  void* b{reinterpret_cast<void*>(2)};
  EXPECT_NO_THROW(
    map.insert(a, 1);
    map.insert(b, 2));

  EXPECT_THROW(map.insert(b, 2), umpire::util::Exception);

  ASSERT_EQ(map.size(), 2);
}

TEST(MemoryMap, find)
{
  umpire::util::MemoryMap<int> map{};

  void* a{reinterpret_cast<void*>(1)};
  void* b{reinterpret_cast<void*>(2)};
  void* c{reinterpret_cast<void*>(3)};
  map.insert(a, 1);
  map.insert(b, 2);

  {
    auto iter{map.find(a)};
    EXPECT_EQ(iter->first, a);
  }

  {
    const auto& cmap{map};
    auto iter{cmap.find(b)};
    EXPECT_EQ(iter->first, b);
  }

  {
    auto iter{map.find(c)};
    EXPECT_EQ(iter, map.end());
    EXPECT_EQ(iter->second, nullptr);
  }
}


TEST(MemoryMap, Iterator)
{
  umpire::util::MemoryMap<int> map{};

  void* a{reinterpret_cast<void*>(1)};
  auto ia{map.insert(a, 1)};

  auto begin{map.begin()};
  ASSERT_EQ(begin->first, a);
  ASSERT_EQ(ia, map.begin());
  ASSERT_EQ(++ia, map.end());
}

TEST(MemoryMap, map_order)
{
  umpire::util::MemoryMap<int> map{};

  void* a{reinterpret_cast<void*>(1)};
  void* b{reinterpret_cast<void*>(2)};
  auto ia{map.insert(a, 1)};
  auto ib{map.insert(b, 2)};
  ASSERT_EQ(++ia, ib);
}
