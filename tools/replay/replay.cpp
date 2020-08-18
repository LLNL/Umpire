//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <ctime>
#include <ratio>
#include <chrono>

#include "umpire/util/Macros.hpp"

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include "umpire/tpl/CLI11/CLI11.hpp"
#include "ReplayInterpreter.hpp"
#include "ReplayMacros.hpp"
#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

int main(int argc, char* argv[])
{
#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
  CLI::App app{"Replay an umpire session from a file"};

  std::string input_file_name;
  app.add_option("-i,--infile", input_file_name,
      "Input file created by Umpire library with UMPIRE_REPLAY=On")
      ->required()
      ->check(CLI::ExistingFile);

  bool time_it{false};
  bool print_stats{false};
  bool print_info{false};
  bool skip_operations{false};

  app.add_flag("-t,--time", time_it, "Display replay times");

  app.add_flag("-s,--stats", print_stats,
      "Dump ULTRA file containing memory usage stats for each Allocator");

  app.add_flag("--info" , print_info,
      "Display information about the replay file");

  app.add_flag("--skip-operations" , skip_operations,
      "Skip Umpire Operations during replays");

  CLI11_PARSE(app, argc, argv);

  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
  std::chrono::duration<double> time_span;

  t1 = std::chrono::high_resolution_clock::now();
  ReplayInterpreter replay(input_file_name);

  replay.buildOperations();

  t2 = std::chrono::high_resolution_clock::now();

  if (time_it) {
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Parsing replay log took " << time_span.count() << " seconds." << std::endl;
  }

  if (print_info) {
    replay.printInfo();
  }

  t1 = std::chrono::high_resolution_clock::now();
  replay.runOperations(print_stats, skip_operations);
  t2 = std::chrono::high_resolution_clock::now();

  if (time_it) {
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Running replay took " << time_span.count() << " seconds." << std::endl;
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
