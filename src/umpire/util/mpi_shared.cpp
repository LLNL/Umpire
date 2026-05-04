//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/mpi_shared.hpp"

#if defined(UMPIRE_ENABLE_MPI)

#include <fstream>
#include <string>

#if defined(__linux__)
#include <sched.h>
#endif

#include "umpire/util/Macros.hpp"
#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/util/numa.hpp"
#endif
#include "umpire/util/error.hpp"

namespace umpire {
namespace util {

namespace {

constexpr int IGNORE_KEY{0};

bool get_socket_color_from_package_id(int& color)
{
#if defined(__linux__)
  const int cpu = sched_getcpu();
  if (cpu < 0) {
    UMPIRE_LOG(Debug, "sched_getcpu failed while determining socket color");
    return false;
  }

  std::ifstream package_id_file{"/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                                "/topology/physical_package_id"};
  if (!package_id_file) {
    UMPIRE_LOG(Debug, "Could not open physical_package_id for cpu " << cpu);
    return false;
  }

  package_id_file >> color;
  if (!package_id_file) {
    UMPIRE_LOG(Debug, "Could not read physical_package_id for cpu " << cpu);
    return false;
  }

  UMPIRE_LOG(Debug, "Using Linux physical_package_id " << color << " for cpu " << cpu);
  return true;
#else
  UMPIRE_USE_VAR(color);
  return false;
#endif
}

bool get_socket_color_from_numa(int& color)
{
#if defined(UMPIRE_ENABLE_NUMA) && defined(__linux__)
  const int cpu = sched_getcpu();
  if (cpu < 0) {
    UMPIRE_LOG(Debug, "sched_getcpu failed while determining NUMA fallback color");
    return false;
  }

  try {
    color = numa::node_of_cpu(cpu);
    UMPIRE_LOG(Debug, "Using NUMA node " << color << " as socket color fallback for cpu " << cpu);
    return true;
  } catch (const std::exception& e) {
    UMPIRE_LOG(Debug, "NUMA fallback failed for cpu " << cpu << ": " << e.what());
    return false;
  }
#else
  UMPIRE_USE_VAR(color);
  return false;
#endif
}

} // end anonymous namespace

MPI_Comm create_shared_communicator(MPI_Comm comm, MemoryResourceTraits::shared_scope scope)
{
  MPI_Comm shared_comm{MPI_COMM_NULL};

  if (scope == MemoryResourceTraits::shared_scope::node) {
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, IGNORE_KEY, MPI_INFO_NULL, &shared_comm);
    return shared_comm;
  }

  if (scope == MemoryResourceTraits::shared_scope::socket) {
    MPI_Comm node_comm{MPI_COMM_NULL};
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, IGNORE_KEY, MPI_INFO_NULL, &node_comm);

    int color{-1};
    if (!get_socket_color_from_package_id(color) && !get_socket_color_from_numa(color)) {
      MPI_Comm_free(&node_comm);
      UMPIRE_ERROR(runtime_error,
                   "Unable to determine a socket color for shared_scope::socket. "
                   "Expected Linux CPU topology data or a NUMA fallback.");
    }

    UMPIRE_LOG(Debug, "Creating socket-scoped shared communicator with color " << color);
    MPI_Comm_split(node_comm, color, IGNORE_KEY, &shared_comm);
    MPI_Comm_free(&node_comm);
    return shared_comm;
  }

  UMPIRE_ERROR(runtime_error,
               fmt::format("Unsupported shared communicator scope: {}", to_string(scope)));
}

} // end of namespace util
} // end of namespace umpire

#endif
