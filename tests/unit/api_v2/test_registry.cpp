//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/detail/registry.hpp"

#include <gtest/gtest.h>

TEST(Registry, SingletonIdentity)
{
  auto& a = umpire::detail::registry::get();
  auto& b = umpire::detail::registry::get();
  EXPECT_EQ(&a, &b);
}

TEST(Registry, IdGeneration)
{
  auto& r = umpire::detail::registry::get();
  const int id0 = r.get_id();
  const int id1 = r.get_id();
  EXPECT_LT(id0, id1);
}

TEST(Registry, AllocationTracking)
{
  auto& r = umpire::detail::registry::get();

  int value = 0;
  umpire::allocation_record rec{&value, sizeof(value), nullptr};

  r.register_allocation(rec);
  EXPECT_TRUE(r.has_allocation(&value));

  auto* found = r.find_allocation(&value);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->ptr, &value);
  EXPECT_EQ(found->size, sizeof(value));

  r.remove_allocation(&value);
  EXPECT_FALSE(r.has_allocation(&value));
  EXPECT_EQ(r.find_allocation(&value), nullptr);
}

