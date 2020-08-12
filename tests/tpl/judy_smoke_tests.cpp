//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include "umpire/tpl/judy/judyL2Array.h"
#include "umpire/tpl/judy/judyLArray.h"

TEST(Judy, LArray)
{
  judyLArray<uint64_t, uint64_t> array;
  array.insert(1, 1);
  array.insert(7, 2);

  uint64_t v = array.find(8);
  ASSERT_EQ(v, 0);

  v = array.find(0);
  ASSERT_EQ(v, 0);

  v = array.find(7);
  ASSERT_EQ(v, 2);

  array.clear();
}

TEST(Judy, L2Array)
{
  using array_t = judyL2Array<uintptr_t, uintptr_t>;
  array_t array;

  array.insert(1, 5);
  array.insert(1, 6);

  array_t::cvector* v = array.find(7);
  ASSERT_TRUE(v == nullptr);

  v = array.find(1);

  ASSERT_EQ(v->size(), 2);
  ASSERT_EQ(v->at(0), 5);
  ASSERT_EQ(v->at(1), 6);

  array.clear();
}
