//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include "umpire/tpl/cxxopts/include/cxxopts.hpp"

TEST(Cxxopts, TestOptions)
{
  const char* args_const[2] = {"cxxopts_smoke_tests", "--test"};
  char** args = const_cast<char**>(args_const);
  int argc = 2;

  cxxopts::Options options(args[0], "test cxxopts");

  options.add_options()("t, test", "test option");

  auto result = options.parse(argc, args);

  ASSERT_TRUE(result.count("test"));
}
