//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef REPLAY_ReplayOptions_HPP
#define REPLAY_ReplayOptions_HPP

#include <string>
#include "umpire/tpl/CLI11/CLI11.hpp"

struct ReplayUsePoolValidator : public CLI::Validator {
  ReplayUsePoolValidator() {
    func_ = [](const std::string &str) {
      if (str != "Quick" && str != "List" && str != "Map") {
        return std::string("Invalid pool name, must be Quick, List, or Map");
      }
      else
        return std::string();
    };
  }
};

struct ReplayUseHeuristicValidator : public CLI::Validator {
  ReplayUseHeuristicValidator() {
    func_ = [](const std::string &str) {
      if (str != "Block" && str != "FreePercentage") {
        return std::string("Invalid heuristic name, must be Block or FreePercentage");
      }
      else
        return std::string();
    };
  }
};

struct ReplayOptions {
  bool time_replay_run{false};    // -t,--time-run
  bool time_replay_parse{false};  // --time-parse
  bool info_only{false};          // --info
  bool dump_statistics{false};    // -d, --dump
  bool track_stats{false};        // -s, --stats
  bool skip_operations{false};    // --skip-operations
  bool force_compile{false};      // -r,--recompile
  bool do_not_demangle{false};    // --no-demangle
  bool quiet{false};              // -q,--quiet
  std::string input_file;         // -i,-infile input_file
  std::string pool_to_use;        // -p,--use-pool
  std::string heuristic_to_use{}; // --use-heuristic
  int heuristic_parm{2};          // --heuristic-parm
};

#endif  // REPLAY_ReplayOptions_HPP

