//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MPI_HPP
#define UMPIRE_MPI_HPP

#if defined(UMPIRE_ENABLE_MPI)
#include "mpi.h"
#endif

namespace umpire {
namespace util {

class MPI {
 public:
  static void initialize(
#if defined(UMPIRE_ENABLE_MPI)
      MPI_Comm umpire_communicator
#endif
  );

  static void finalize();

  static int getRank();
  static int getSize();

  static void sync();

  static void logMpiInfo();

  static bool isInitialized();

 private:
  static int s_rank;
  static int s_world_size;

  static bool s_initialized;

  static int s_mpi_init_called;

#if defined(UMPIRE_ENABLE_MPI)
  static MPI_Comm s_communicator;
#endif
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_MPI_HPP
