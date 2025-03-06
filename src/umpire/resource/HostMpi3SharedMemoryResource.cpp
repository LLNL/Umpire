//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/HostMpi3SharedMemoryResource.hpp"

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

HostMpi3SharedMemoryResource::HostMpi3SharedMemoryResource(const std::string& name, int id, MemoryResourceTraits traits)
    : MemoryResource{name, id, traits}
{
  constexpr int IGNORE_KEY{0};
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, IGNORE_KEY, MPI_INFO_NULL, &m_shared_comm);
  MPI_Comm_rank(m_shared_comm, &m_local_rank);
}

HostMpi3SharedMemoryResource::~HostMpi3SharedMemoryResource()
{
  // TODO: Add finalize routine for cleanup pre MPI_Finalize
  // MPI_Comm_free(&m_shared_comm);
}

void* HostMpi3SharedMemoryResource::allocate(std::size_t bytes)
{
  void* ptr;
  MPI_Win win;
  MPI_Aint local_size = (m_local_rank != 0) ? 0 : bytes;
  MPI_Aint size = bytes;
  int disp{sizeof(unsigned char)};

  MPI_Win_allocate_shared(local_size, disp, MPI_INFO_NULL, m_shared_comm, &ptr, &win);
  MPI_Win_shared_query(win, 0, &size, &disp, &ptr);
  m_shared_windows[ptr] = win;

  return ptr;
}

void HostMpi3SharedMemoryResource::deallocate(void* ptr, std::size_t)
{
  auto window = m_shared_windows.find(ptr);
  if (window != m_shared_windows.end()) {
    MPI_Win_free(&(window->second));
    m_shared_windows.erase(window);
  } else {
    UMPIRE_ERROR(umpire::unknown_pointer_error, "");
  }
}

bool HostMpi3SharedMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if (p == Platform::host)
    return true;
  else // TODO: check this
    return false;
}

Platform HostMpi3SharedMemoryResource::getPlatform() noexcept
{
  return Platform::host;
}

} // end of namespace resource
} // end of namespace umpire
