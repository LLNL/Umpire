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

#include "umpire/Umpire.hpp"
#include "umpire/util/IOManager.hpp"
#include "umpire/ResourceManager.hpp"

#if defined(UMPIRE_ENABLE_MPI)
#include "mpi.h"
#endif

int main(int argc, char** argv) {
#if defined(UMPIRE_ENABLE_MPI)
  MPI_Init(&argc, &argv);
#else
  (void) argc;
  (void) argv;
#endif

  auto& rm = umpire::ResourceManager::getInstance();

  umpire::log << "testing log stream" << std::endl;
  umpire::replay << "testing replay stream" << std::endl;
  umpire::error << "testing error stream" << std::endl;

  (void) rm;

#if defined(UMPIRE_ENABLE_MPI)
  MPI_Finalize();
#endif

  return 0;
}
