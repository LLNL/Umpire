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

TEST(MemoryMap, InsertAndSize)
{
  umpire::util::MemoryMap<int> map{};

  void* a{reinterpret_cast<void*>(1)};
  void* b{reinterpret_cast<void*>(2)};
  EXPECT_NO_THROW(
    map.insert(a, 1);
    map.insert(b, 2));

  ASSERT_EQ(map.size(), 2);
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

TEST(MemoryMap, Ordering)
{
  umpire::util::MemoryMap<int> map{};

  void* a{reinterpret_cast<void*>(1)};
  void* b{reinterpret_cast<void*>(2)};
  auto ia{map.insert(a, 1)};
  auto ib{map.insert(b, 2)};
  ASSERT_EQ(++ia, ib);
}
