//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <ratio>
#include <string>

#include "umpire/util/Macros.hpp"

#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
#include "umpire/tpl/CLI11/CLI11.hpp"
#include "ReplayInterpreter.hpp"
#include "ReplayMacros.hpp"
#include "ReplayOptions.hpp"

const static ReplayUsePoolValidator ReplayValidPool;

#endif // !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)

int main(int argc, char* argv[])
{
#if !defined(_MSC_VER) && !defined(_LIBCPP_VERSION)
  ReplayOptions options;
  CLI::App app{"Replay an umpire session from a file"};

  app.add_option("-i,--infile", options.input_file,
      "Input file created by Umpire library with UMPIRE_REPLAY=On")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_flag("-q,--quiet", options.quiet,
        "Only errors will be displayed.");

  app.add_flag("-t,--time-run", options.time_replay_run,
        "Display time information for replay running operations");

  app.add_flag("-s,--stats", options.print_statistics,
      "Dump ULTRA file containing memory usage stats for each Allocator");

  app.add_flag("--size-stats", options.print_size_stats,
      "Display pool allocaiton size statistics");

  app.add_flag("--info-only" , options.info_only,
      "Information about replay file, no actual replay performed");

  app.add_flag("--no-demangle" , options.do_not_demangle,
      "Disable demangling of replay file");

  app.add_flag("--skip-operations" , options.skip_operations,
      "Skip Umpire Operations during replays");

  app.add_flag("-r,--recompile" , options.force_compile,
      "Force recompile replay binary");

  app.add_option("-p,--use-pool", options.pool_to_use,
    "Specify pool to use: List, Map, or Quick")->check(ReplayValidPool);

  CLI11_PARSE(app, argc, argv);

  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
  std::chrono::duration<double> time_span;

  t1 = std::chrono::high_resolution_clock::now();
  ReplayInterpreter replay(options);

  replay.buildOperations();

  if (options.time_replay_parse) {
    t2 = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Parsing replay log took " << time_span.count() << " seconds." << std::endl;
  }

  if ( !options.info_only ) {
    t1 = std::chrono::high_resolution_clock::now();
    replay.runOperations();

    if (options.time_replay_run) {
      t2 = std::chrono::high_resolution_clock::now();
      time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
      std::cout << "Running replay took " << time_span.count() << " seconds." << std::endl;
    }
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
