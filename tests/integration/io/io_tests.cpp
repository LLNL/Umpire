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
#include "umpire/util/Logger.hpp"

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

  if (enable_logging) {
#if defined(_MSC_VER)
    _putenv_s("UMPIRE_LOG_LEVEL", "Info");
#else
    setenv("UMPIRE_LOG_LEVEL", "Info", 1);
#endif
    umpire::util::Logger::initialize();
    umpire::util::Logger::log(umpire::util::message::Info, "testing log stream", __FILE__, __LINE__);
    umpire::util::Logger::finalize();
  }

  std::cerr << "testing error stream" << std::endl;

#if defined(UMPIRE_ENABLE_MPI)
  MPI_Finalize();
#endif

  return 0;
}
