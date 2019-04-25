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

#include "gtest/gtest.h"

#include "umpire/tpl/cxxopts/include/cxxopts.hpp"

TEST(Cxxopts, )
{
  const char* args_const[2] = {"cxxopts_smoke_tests", "--test"};
  char** args = const_cast<char**>(args_const);
  int argc = 2;

  cxxopts::Options options(args[0], "test cxxopts");

  options.add_options()("t, test", "test option");

  auto result = options.parse(argc, args);

  ASSERT_TRUE(result.count("test"));
}
