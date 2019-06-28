//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
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
  static void initialize();
  static void finalize();

  static int getRank();
  static int getSize();
private:
  static int s_rank;
  static int s_world_size;

  static bool s_initialized;
};

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_MPI_HPP
