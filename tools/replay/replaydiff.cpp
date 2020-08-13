//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
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
#include "umpire/tpl/CLI11/CLI11.hpp"
#include "ReplayInterpreter.hpp"
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

int main(int argc, char* argv[])
{
#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
  CLI::App app{"Compare two replay result files created"
                " by Umpire library with UMPIRE_REPLAY=On"};

  std::vector<std::string> positional_args;

  app.add_option("files", positional_args, "replay_file_1 replay_file_2")
    ->required()
    ->expected(2)
    ->check(CLI::ExistingFile);

  CLI11_PARSE(app, argc, argv);

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
    << "However, this program was compiled with -stdlib=libc++ which does " << std::endl
    << "not have this feature." << std::endl;
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

  return 0;
}
