//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <ratio>
#include <string>
#include <vector>

#include "umpire/util/Macros.hpp"

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include "umpire/tpl/cxxopts/include/cxxopts.hpp"
#include "ReplayInterpreter.hpp"
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

int main(int argc, char* argv[])
{
#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
  std::vector<std::string> positional_args;

  cxxopts::Options options(argv[0], "Compare two replay result files created"
    " by Umpire library with UMPIRE_REPLAY=On");
  options
    .positional_help("replay_file_1 replay_file_2")
    .show_positional_help();

  options.add_options()
    ("h, help", "Print help");

  options.add_options()
    ("positional_args", "Positional parameters",
        cxxopts::value<std::vector<std::string>>(positional_args));

  std::vector<std::string> pos_names = {"positional_args"};

  options.parse_positional(pos_names.begin(), pos_names.end());

  auto command_line_args = options.parse(argc, argv);

  if (command_line_args.count("help")) {
    std::cout << options.help({""}) << std::endl;
    exit(0);
  }

  if (positional_args.size() != 2) {
    std::cout << options.help({""}) << std::endl;
    exit(1);
  }

  const std::string& left_filename = positional_args[0];
  const std::string& right_filename = positional_args[1];

  ReplayInterpreter left{left_filename};
  ReplayInterpreter right{right_filename};

  left.buildOperations();
  right.buildOperations();

  if (left.compareOperations(right) == false) {
    std::cerr << "Miscompare:" << std::endl;
    return -2;
  }
#else
  UMPIRE_USE_VAR(argc);
  UMPIRE_USE_VAR(argv);
  std::cerr << "This program requires the ability to demangle C++" << std::endl
    << "However, this program was compiled with stdlib=c++ which does " << std::endl
    << "not have this feature." << std::endl;
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

  return 0;
}
