//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "umpire/tpl/CLI11/CLI11.hpp"

static std::string getVersion()
{
  std::stringstream sstr;

  sstr << CLI11_VERSION;

  return sstr.str();
}

static int test_parse_fun(int argc, const char** argv)
{
  std::stringstream sstr;
  sstr << "Smoke test for CLI11 TPL, version " << getVersion();

  CLI::App app{sstr.str()};

  // variables for command line options
  bool opt_bool{false};
  app.add_flag("-b,--some-bool,!--no-some-bool", opt_bool, "boolean flag")
      ->capture_default_str();

  int opt_int{0};
  app.add_option("-i,--some-int", opt_int, "integer input")->required();

  float opt_float{1.0};
  app.add_option("-f,--some-float", opt_float, "float input")
      ->capture_default_str()
      ->check(CLI::Range(1., 4.).description("Range [1,4]"));

  std::string opt_str;
  app.add_option("-s,--some-string", opt_str, "string input");

  app.get_formatter()->column_width(40);

  // Output information about command line options
  // some-bool is always available
  std::cout << "Boolean input was '" << std::boolalpha << opt_bool << "'"
            << std::endl;

  // some-int is always available
  std::cout << "Integer input was '" << opt_int << "'" << std::endl;

  // some-float is available sometimes
  if (app.count("--some-float")) {
    std::cout << "Float input was '" << opt_float << "'" << std::endl;
  }

  // some-string is available sometimes
  if (app.count("--some-string")) {
    std::cout << "String input was '" << opt_str << "'" << std::endl;
  }

  CLI11_PARSE(app, argc, argv);

  EXPECT_TRUE(opt_bool);
  EXPECT_EQ(opt_int, 42);
  opt_float *= 100;
  int ifloat = static_cast<int>(opt_float);
  EXPECT_EQ(ifloat, 314);
  EXPECT_EQ(opt_str, std::string("hello world"));

  return 0;
}

//-----------------------------------------------------------------------------
TEST(CLI11, Parsing)
{
  int argc{7};
  const char* argv[] = {
      "cli11_smoke_test", "-b",         "-i", "42", "--some-float=3.14",
      "--some-string",    "hello world"};

  ASSERT_EQ(0, test_parse_fun(argc, argv));
}
