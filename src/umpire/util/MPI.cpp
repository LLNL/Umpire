//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/MPI.hpp"

#include "umpire/Replay.hpp"
#include "umpire/config.hpp"
#include "umpire/util/Macros.hpp"

#if defined(UMPIRE_ENABLE_MPI)
#include "mpi.h"
#endif

namespace umpire {
namespace util {

int MPI::s_rank = -1;
int MPI::s_world_size = -1;
bool MPI::s_initialized = false;
int MPI::s_mpi_init_called = 0;
#if defined(UMPIRE_ENABLE_MPI)
MPI_Comm MPI::s_communicator = MPI_COMM_NULL;
#endif

void MPI::initialize(
#if defined(UMPIRE_ENABLE_MPI)
    MPI_Comm comm
#endif
)
{
  if (!s_initialized) {
#if !defined(UMPIRE_ENABLE_MPI)
    s_rank = 0;
    s_world_size = 1;
    s_mpi_init_called = 1;
    s_initialized = true;
#else
    MPI_Initialized(&s_mpi_init_called);

    if (s_mpi_init_called) {
      s_communicator = comm;
      MPI_Comm_rank(s_communicator, &s_rank);
      MPI_Comm_size(s_communicator, &s_world_size);
      s_initialized = true;
    }

#endif
  } else {
    UMPIRE_ERROR(
        "umpire::MPI already initialized, cannot call initialize() again!");
  }
}

void MPI::finalize()
{
  if (s_initialized) {
#if !defined(UMPIRE_ENABLE_MPI)
    s_rank = -1;
    s_world_size = -1;
#endif
  } else {
    UMPIRE_ERROR(
        "Cannot call MPI::finalize() when umpire::MPI not initialiazed");
  }
}

int MPI::getRank()
{
  if (!s_initialized) {
    UMPIRE_LOG(Warning,
               "umpire::MPI not initialized, returning rank=" << s_rank);
  }

  return s_rank;
}

int MPI::getSize()
{
  if (!s_initialized) {
    UMPIRE_LOG(Warning,
               "umpire::MPI not initialized, returning size=" << s_world_size);
  }

  return s_world_size;
}

void MPI::sync()
{
  if (s_initialized) {
#if defined(UMPIRE_ENABLE_MPI)
    MPI_Barrier(s_communicator);
#endif
  } else {
    UMPIRE_ERROR("Cannot call MPI::sync() before umpire::MPI is initialized");
  }
}

void MPI::logMpiInfo()
{
  if (s_initialized) {
#if defined(UMPIRE_ENABLE_MPI)
    UMPIRE_LOG(Info, "MPI rank: " << s_rank);
    UMPIRE_LOG(Info, "MPI comm size: " << s_world_size);

    UMPIRE_REPLAY("\"event\": \"mpi\", \"payload\": { \"rank\":"
                  << s_rank << ", \"size\":" << s_world_size << "}");
#endif
  }
}

bool MPI::isInitialized()
{
  return s_initialized;
}

} // end of namespace util
} // end of namespace umpire
