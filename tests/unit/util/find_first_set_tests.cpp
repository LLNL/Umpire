//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <limits>
#include <random>

#include "gtest/gtest.h"
#include "umpire/util/find_first_set.hpp"

using umpire::util::find_first_set;

TEST(FindFirstSet, ZeroBits)
{
  const auto bit = find_first_set(0);
  EXPECT_EQ(bit, 0);
}

TEST(FindFirstSet, OneBit)
{
  // include sign bit
  constexpr int n_bits = std::numeric_limits<int>::digits + 1;

  for (int index = 0; index < n_bits; ++index) {
    const int mask = 1 << index;
    const auto bit = find_first_set(mask);
    EXPECT_EQ(bit, index + 1);
  }
}

TEST(FindFirstSet, MultipleBits)
{
  // include sign bit
  constexpr int n_bits = std::numeric_limits<int>::digits + 1;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, n_bits - 1);

  constexpr int n_iter = 4;
  auto min_index = n_bits;
  int mask = 0;

  for (int iter = 0; iter < n_iter; ++iter) {
    const auto index = dist(gen);
    const int m = 1 << index;
    mask |= m;
    min_index = std::min(min_index, index);
  }

  const auto bit = find_first_set(mask);
  EXPECT_EQ(bit, min_index + 1);
}
