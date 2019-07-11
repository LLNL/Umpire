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

#include "umpire/tpl/cxxopts/include/cxxopts.hpp"
#include "util/Replay.hpp"

int main(int argc, char* argv[])
{
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

  Replay left(left_filename);
  Replay right(right_filename);

  while (1) {
    std::string left_raw, left_sym;
    std::string right_raw, right_sym;

    int left_rval = left.symbol_op(left_raw, left_sym);
    int right_rval = right.symbol_op(right_raw, right_sym);

    if (!left_rval && !right_rval) {
      break;  // Both at EOF, we are done
    }
    else if (!left_rval || !right_rval) {
      std::cerr << "Size mismatch" << std::endl;
      return -1;
    }
    else if (left_rval != right_rval) {
      std::cerr << "Error in reading input from one of the compared files" << std::endl;
      return -1;
    }

    if (left_sym != right_sym) {
      std::cerr << "Miscompare:" << std::endl;
      std::cerr << left_raw << std::endl;
      std::cerr << right_raw << std::endl;
      return -2;
    }
    // std::cout << "OK: " << left_sym;
  }

  return 0;
}
