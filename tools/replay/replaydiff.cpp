//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
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
#include "ReplayInterpreter.hpp"
#include "ReplayOptions.hpp"
#include "umpire/CLI11/CLI11.hpp"
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

int main(int argc, char* argv[])
{
#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
  ReplayOptions lhs_options;
  ReplayOptions rhs_options;
  CLI::App app{"Compare two replay result files created"
                " by Umpire library with UMPIRE_REPLAY=On"};

  std::vector<std::string> positional_args;

  app.add_flag("-r,--recompile" , lhs_options.force_compile,
      "Force recompile replay binary");

  app.add_flag("-q,--quiet", lhs_options.quiet,
        "Only errors will be displayed.");

  app.add_option("files", positional_args, "replay_file_1 replay_file_2")
    ->required()
    ->expected(2)
    ->check(CLI::ExistingFile);

  CLI11_PARSE(app, argc, argv);

  rhs_options = lhs_options;

  lhs_options.input_file = positional_args[0];
  rhs_options.input_file = positional_args[1];

  ReplayInterpreter left{lhs_options};
  ReplayInterpreter right{rhs_options};

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
