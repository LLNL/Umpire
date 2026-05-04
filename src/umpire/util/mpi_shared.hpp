//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_mpi_shared_HPP
#define UMPIRE_mpi_shared_HPP

#include "umpire/config.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

#if defined(UMPIRE_ENABLE_MPI)
#include "mpi.h"
#endif

namespace umpire {
namespace util {

#if defined(UMPIRE_ENABLE_MPI)
MPI_Comm create_shared_communicator(MPI_Comm comm, MemoryResourceTraits::shared_scope scope);
#endif

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_mpi_shared_HPP
