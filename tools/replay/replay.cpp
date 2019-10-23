//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <ctime>
#include <ratio>
#include <chrono>

#include "umpire/tpl/cxxopts/include/cxxopts.hpp"
#include "ReplayInterpreter.hpp"
#include "ReplayMacros.hpp"

int main(int argc, char* argv[])
{
  cxxopts::Options options(argv[0], "Replay an umpire session from a file");

  options
    .add_options()
    (  "h, help"
     , "Print help"
    )
    (  "a, allocation_map"
     , "Replay allocation map"
    )
    (  "t, time"
     , "Display replay times"
    )
    (  "i, infile"
     , "Input file created by Umpire library with UMPIRE_REPLAY=On"
     , cxxopts::value<std::string>(), "FILE"
    )
    (  "c, compile"
     , "Compile input file to binary format for faster replays"
    )
    (
      "s, stats"
      , "Dump ULTRA file containing memory usage stats for each Allocator"
    )
  ;

  auto command_line_args = options.parse(argc, argv);

  if (command_line_args.count("help")) {
    std::cout << options.help({""}) << std::endl;
    exit(0);
  }

  if ( ! command_line_args.count("infile") )
    REPLAY_ERROR("No input file specified");

  std::string input_file_name = command_line_args["infile"].as<std::string>();

  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
  std::chrono::duration<double> time_span;

  t1 = std::chrono::high_resolution_clock::now();
  ReplayInterpreter replay(input_file_name);

  if (command_line_args.count("compile")) {
    replay.compile();
  }
  else if (command_line_args.count("allocation_map")) {
    replay.buildAllocMapOperations();
  }
  else {
    replay.buildOperations();
  }

  t2 = std::chrono::high_resolution_clock::now();

  if (command_line_args.count("time")) {
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Parsing replay log took " << time_span.count() << " seconds." << std::endl;
  }

  t1 = std::chrono::high_resolution_clock::now();
  const bool print_statistics{command_line_args.count("stats") > 0};
  replay.runOperations(print_statistics);
  t2 = std::chrono::high_resolution_clock::now();

  if (command_line_args.count("time")) {
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Running replay took " << time_span.count() << " seconds." << std::endl;
  }

  return 0;
}
