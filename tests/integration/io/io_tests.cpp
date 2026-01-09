//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/util/MPI.hpp"
#include "umpire/util/io.hpp"

#if defined(UMPIRE_ENABLE_MPI)
#include "mpi.h"
#endif

#include "umpire/CLI11/CLI11.hpp"

int main(int argc, char** argv)
{
#if defined(UMPIRE_ENABLE_MPI)
  MPI_Init(&argc, &argv);
  umpire::initialize(MPI_COMM_WORLD);
#else
  (void)argc;
  (void)argv;
  umpire::initialize();
#endif

  bool enable_logging{false};
  CLI::App app{"IO tests"};

  app.add_flag("-l,--enable-logging", enable_logging, "Enable logging output");

  CLI11_PARSE(app, argc, argv);

  umpire::util::initialize_io(enable_logging);

  umpire::log() << "testing log stream" << std::endl;
  umpire::error() << "testing error stream" << std::endl;

#if defined(UMPIRE_ENABLE_MPI)
  MPI_Finalize();
#endif

  return 0;
}
