//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <string>

#include "umpire/tpl/CLI11/CLI11.hpp"

std::string getVersion()
{
  std::stringstream sstr;

  sstr << CLI11_VERSION;

  return sstr.str();
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  std::stringstream sstr;
  sstr << "Smoke test for CLI11 TPL, version " << getVersion();

  CLI::App app{sstr.str()};

  // variables for command line options
  bool opt_bool = false;
  int opt_int;
  float opt_float = 1.0;
  std::string opt_str;

  // Add the command line options
  app.add_flag("-b,--some-bool,!--no-some-bool", opt_bool, "boolean flag")
      ->capture_default_str();

  app.add_option("-i,--some-int", opt_int, "integer input")->required();

  app.add_option("-f,--some-float", opt_float, "float input")
      ->capture_default_str()
      ->check(CLI::Range(1., 4.).description("Range [1,4]"));

  app.add_option("-s,--some-string", opt_str, "string input");

  app.get_formatter()->column_width(40);

  CLI11_PARSE(app, argc, argv);

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

  return 0;
}
