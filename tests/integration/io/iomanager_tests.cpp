//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/util/MPI.hpp"
#include "umpire/util/IOManager.hpp"

#if defined(UMPIRE_ENABLE_MPI)
#include "mpi.h"
#endif

#include "umpire/tpl/cxxopts/include/cxxopts.hpp"

int main(int argc, char** argv) {
#if defined(UMPIRE_ENABLE_MPI)
  MPI_Init(&argc, &argv);
#else
  (void) argc;
  (void) argv;
#endif

  cxxopts::Options options(argv[0], "Replay an umpire session from a file");

  options
    .add_options()
    (  "l, enable-logging"
     , "Enable logging output"
    )
    (  "r, enable-replay"
     , "Enable replay output"
    );

  auto result = options.parse(argc, argv);

  bool enable_logging = false;
  bool enable_replay = false;

  if (result.count("enable-logging")) {
    enable_logging = true;
  }

  if (result.count("enable-replay")) {
    enable_replay = true;
  }

  umpire::util::MPI::initialize();
  umpire::util::IOManager::initialize(enable_logging, enable_replay);

  umpire::log << "testing log stream" << std::endl;
  umpire::replay << "testing replay stream" << std::endl;
  umpire::error << "testing error stream" << std::endl;

#if defined(UMPIRE_ENABLE_MPI)
  MPI_Finalize();
#endif

  return 0;
}
