//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include <cstddef>

#include "umpire/allocation_record.hpp"

TEST(allocation_record, ConstructAndAccess)
{
  int value{0};
  void* ptr = &value;
  constexpr std::size_t size{64};
  umpire::memory* strategy{nullptr};

  umpire::allocation_record record{ptr, size, strategy};
  EXPECT_EQ(record.ptr, ptr);
  EXPECT_EQ(record.size, size);
  EXPECT_EQ(record.strategy, strategy);
}

TEST(allocation_record, DefaultConstruct)
{
  umpire::allocation_record record{};
  EXPECT_EQ(record.ptr, nullptr);
  EXPECT_EQ(record.size, 0);
  EXPECT_EQ(record.strategy, nullptr);
}
