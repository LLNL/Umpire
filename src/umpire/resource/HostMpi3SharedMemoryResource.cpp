//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/HostMpi3SharedMemoryResource.hpp"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/MPI.hpp"
#include "umpire/util/Macros.hpp"
#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/util/numa.hpp"
#endif
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

namespace {

constexpr int IGNORE_KEY{0};

std::string get_mpi_error_message(int error_code)
{
  char buffer[MPI_MAX_ERROR_STRING];
  int length{0};
  const int status = MPI_Error_string(error_code, buffer, &length);

  if (status != MPI_SUCCESS) {
    return fmt::format("MPI error code {} (MPI_Error_string failed with code {})", error_code, status);
  }

  return std::string{buffer, static_cast<std::size_t>(length)};
}

void check_mpi_call(int error_code, const char* call_name)
{
  if (error_code != MPI_SUCCESS) {
    UMPIRE_ERROR(runtime_error, fmt::format("{} failed: {}", call_name, get_mpi_error_message(error_code)));
  }
}

//////////
// Start of linux defined Socket support
//////////

#if defined(__linux__)
/*!
 * \brief Map an operating-system CPU index to the socket color used for
 * communicator splitting.
 *
 * The helper first reads Linux topology information and then falls back to the
 * NUMA helper when available.
 *
 * \param cpu CPU index from the current rank's affinity mask.
 * \param color Output socket identifier used as the MPI_Comm_split color when
 *        the lookup succeeds.
 *
 * \return true when a socket identifier was found for \p cpu, false otherwise.
 */
bool try_get_socket_color_for_cpu(int cpu, int& color)
{
  std::ifstream package_id_file{"/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/physical_package_id"};
  if (package_id_file) {
    package_id_file >> color;
    if (package_id_file) {
      UMPIRE_LOG(Debug, "Using Linux physical_package_id " << color << " for cpu " << cpu);
      return true;
    }

    UMPIRE_LOG(Debug, "Read package ID for file but could not read physical_package_id for cpu " << cpu);
  } else {
    UMPIRE_LOG(Debug, "Could not open physical_package_id for cpu " << cpu);
  }

#if defined(UMPIRE_ENABLE_NUMA)
  try {
    color = numa::node_of_cpu(cpu);
    UMPIRE_LOG(Debug, "Using NUMA node " << color << " as socket color fallback for cpu " << cpu);
    return true;
  } catch (const std::exception& e) {
    UMPIRE_LOG(Debug, "NUMA fallback failed for cpu " << cpu << ": " << e.what());
  }
#endif

  UMPIRE_USE_VAR(cpu);
  UMPIRE_USE_VAR(color);
  return false;
}

/*!
 * \brief Format the set of socket colors found in a rank's affinity mask.
 *
 * \param colors Socket identifiers collected while inspecting the affinity
 *        mask.
 *
 * \return Comma-separated list used in diagnostic messages.
 */
std::string format_socket_colors(const std::set<int>& colors)
{
  std::ostringstream ss;

  bool first = true;
  for (int color : colors) {
    if (!first) {
      ss << ", ";
    }
    ss << color;
    first = false;
  }

  return ss.str();
}

bool try_get_socket_color_from_affinity(int& socket_color, std::string& reason)
{
  const long cpu_count_long = sysconf(_SC_NPROCESSORS_CONF);
  if (cpu_count_long <= 0) {
    reason = "Could not determine configured CPU count (sysconf(_SC_NPROCESSORS_CONF) failed)";
    return false;
  }

  if (cpu_count_long > std::numeric_limits<int>::max()) {
    reason = "Configured CPU count exceeds supported range";
    return false;
  }

  const int cpu_count = static_cast<int>(cpu_count_long);

  const auto set_size = CPU_ALLOC_SIZE(cpu_count);
  cpu_set_t* cpu_set = CPU_ALLOC(cpu_count);
  if (!cpu_set) {
    reason = "Failed to allocate CPU affinity mask";
    return false;
  }

  CPU_ZERO_S(set_size, cpu_set);

  if (sched_getaffinity(0, set_size, cpu_set) != 0) {
    reason = fmt::format("sched_getaffinity failed: {}", std::strerror(errno));
    CPU_FREE(cpu_set);
    return false;
  }

  std::set<int> socket_colors;
  bool saw_cpu = false;

  for (int cpu = 0; cpu < cpu_count; ++cpu) {
    if (!CPU_ISSET_S(cpu, set_size, cpu_set)) {
      continue;
    }
    saw_cpu = true;

    int color{-1};
    if (!try_get_socket_color_for_cpu(cpu, color)) {
      reason = "Unable to determine socket identifier for a CPU in this rank's affinity mask";
      CPU_FREE(cpu_set);
      return false;
    }

    socket_colors.insert(color);
    if (socket_colors.size() > 1) {
      reason =
          fmt::format("Socket-scoped MPI3 shared memory requires each rank pinned to exactly one socket; this rank spans "
                      "multiple sockets ({})",
                      format_socket_colors(socket_colors));
      CPU_FREE(cpu_set);
      return false;
    }
  }

  CPU_FREE(cpu_set);

  if (!saw_cpu) {
    reason = "Rank CPU affinity mask is empty";
    return false;
  }

  if (socket_colors.empty()) {
    reason = "No socket identifiers found in affinity mask";
    return false;
  }

  socket_color = *socket_colors.begin();
  return true;
}
#else
bool try_get_socket_color_from_affinity(int& socket_color, std::string& reason)
{
  UMPIRE_USE_VAR(socket_color);
  reason = "Socket-scoped MPI3 shared memory requires Linux CPU affinity information";
  return false;
}
#endif
//////////
// End of linux defined Socket support
//////////

/*!
 * \brief Build the communicator that defines which ranks share allocations.
 *
 * \param comm Parent communicator used to discover ranks that may share host
 *        memory.
 * \param scope Requested sharing scope from MemoryResourceTraits. Node scope
 *        keeps one communicator per node, while socket scope further splits the
 *        node-local communicator using socket affinity.
 *
 * \return Communicator containing exactly the ranks that participate in each
 * shared-memory allocation for the resource.
 */
MPI_Comm create_shared_communicator(MPI_Comm comm, MemoryResourceTraits::shared_scope scope)
{
  MPI_Comm shared_comm{MPI_COMM_NULL};

  if (scope == MemoryResourceTraits::shared_scope::node) {
    check_mpi_call(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, IGNORE_KEY, MPI_INFO_NULL, &shared_comm),
                   "MPI_Comm_split_type");
    return shared_comm;
  }

  if (scope == MemoryResourceTraits::shared_scope::socket) {
    MPI_Comm node_comm{MPI_COMM_NULL};
    check_mpi_call(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, IGNORE_KEY, MPI_INFO_NULL, &node_comm),
                   "MPI_Comm_split_type");

    int color{-1};
    std::string reason;
    const int local_affinity_valid = try_get_socket_color_from_affinity(color, reason) ? 1 : 0;
    int all_affinity_valid{0};

    const int allreduce_status = MPI_Allreduce(&local_affinity_valid, &all_affinity_valid, 1, MPI_INT, MPI_MIN, comm);
    if (allreduce_status != MPI_SUCCESS) {
      MPI_Comm_free(&node_comm);
      check_mpi_call(allreduce_status, "MPI_Allreduce");
    }

    if (!all_affinity_valid) {
      check_mpi_call(MPI_Comm_free(&node_comm), "MPI_Comm_free");
      if (local_affinity_valid) {
        reason = "Another rank in the parent communicator could not determine a single socket affinity";
      }
      UMPIRE_ERROR(runtime_error, reason);
    }

    UMPIRE_LOG(Debug, "Creating socket-scoped shared communicator with color " << color);

    const int split_status = MPI_Comm_split(node_comm, color, IGNORE_KEY, &shared_comm);
    if (split_status != MPI_SUCCESS) {
      if (node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&node_comm);
      }
      check_mpi_call(split_status, "MPI_Comm_split");
    }

    check_mpi_call(MPI_Comm_free(&node_comm), "MPI_Comm_free");

    return shared_comm;
  }

  UMPIRE_ERROR(runtime_error,
               fmt::format("Unsupported shared communicator scope: {}", to_string(scope)));
}

} // end anonymous namespace

bool affinity_maps_to_single_socket(std::string& reason)
{
  int socket_color{-1};
  return try_get_socket_color_from_affinity(socket_color, reason);
}

HostMpi3SharedMemoryResource::HostMpi3SharedMemoryResource(const std::string& name, int id, MemoryResourceTraits traits)
    : MemoryResource{name, id, traits}
{
  m_shared_comm = create_shared_communicator(util::MPI::getCommunicator(), traits.scope);
  check_mpi_call(MPI_Comm_rank(m_shared_comm, &m_local_rank), "MPI_Comm_rank");

  // Free the comm at exit during cleanup in MPI_Finalize. We pass the m_shared_comm
  // by turning it into an int (as for Fortran) and then decoding that in the callback.
  int keyval = 0;
  check_mpi_call(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, free_comm, &keyval, nullptr), "MPI_Comm_create_keyval");
  check_mpi_call(MPI_Comm_set_attr(MPI_COMM_SELF, keyval, (void*)(intptr_t)MPI_Comm_c2f(m_shared_comm)),
                 "MPI_Comm_set_attr");
}

HostMpi3SharedMemoryResource::~HostMpi3SharedMemoryResource()
{
  // NOTE: m_shared_comm is freed at cleanup pre MPI_Finalize
}

void* HostMpi3SharedMemoryResource::allocate(std::size_t bytes)
{
  void* ptr{nullptr};
  MPI_Win win{MPI_WIN_NULL};
  MPI_Aint local_size = (m_local_rank != 0) ? 0 : bytes;
  MPI_Aint size = bytes;
  int disp{sizeof(unsigned char)};

  check_mpi_call(MPI_Win_allocate_shared(local_size, disp, MPI_INFO_NULL, m_shared_comm, &ptr, &win),
                 "MPI_Win_allocate_shared");
  check_mpi_call(MPI_Win_shared_query(win, 0, &size, &disp, &ptr), "MPI_Win_shared_query");
  m_shared_windows[ptr] = win;

  return ptr;
}

void HostMpi3SharedMemoryResource::deallocate(void* ptr, std::size_t)
{
  auto window = m_shared_windows.find(ptr);
  if (window != m_shared_windows.end()) {
    check_mpi_call(MPI_Win_free(&(window->second)), "MPI_Win_free");
    m_shared_windows.erase(window);
  } else {
    UMPIRE_ERROR(umpire::unknown_pointer_error, "");
  }
}

bool HostMpi3SharedMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if (p == Platform::host)
    return true;
  else
    return false;
}

Platform HostMpi3SharedMemoryResource::getPlatform() noexcept
{
  return Platform::host;
}

MPI_Comm HostMpi3SharedMemoryResource::getSharedCommunicator() const noexcept
{
  return m_shared_comm;
}

int HostMpi3SharedMemoryResource::free_comm(MPI_Comm UMPIRE_UNUSED_ARG(comm), int UMPIRE_UNUSED_ARG(keyval),
                                            void* attribute_val, void* UMPIRE_UNUSED_ARG(extra_state))
{
  // Interpret attribute_val as a MPI_Fint comm number.
  const auto comm_number = (MPI_Fint)(intptr_t)(attribute_val);
  MPI_Comm comm_to_free = MPI_Comm_f2c(comm_number);
  return MPI_Comm_free(&comm_to_free);
}

} // end of namespace resource
} // end of namespace umpire
