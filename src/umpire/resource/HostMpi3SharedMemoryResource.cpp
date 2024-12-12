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

HostMpi3SharedMemoryResource::HostMpi3SharedMemoryResource(Platform platform, const std::string& name, int id,
                                                   MemoryResourceTraits traits)
    : MemoryResource{name, id, traits}, m_platform{platform}, pimpl{new impl{name, traits.size}}
{
  MPI_comm_split(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, IGNORE_KEY, MPI_INFO_NULL, &m_shared_comm);
  MPI_comm_rank(m_shared_comm, &m_local_rank);
}

HostMpi3SharedMemoryResource::~HostMpi3SharedMemoryResource()
{
  MPI_Comm_free(m_shared_comm);
}

void* HostMpi3SharedMemoryResource::allocate(std::size_t bytes)
{
  void* ptr;
  MPI_Win win;
  MPI_Aint size = (m_local_rank != 0) ? 0 : bytes;
  int disp{sizeof(char)};

  MPI_Win_allocate_shared(size, disp, MPI_INFO_NULL, m_shared_comm, &ptr, &win)
  m_shared_windows[ptr] = win;

  return ptr;
}

void HostMpi3SharedMemoryResource::deallocate(void* ptr, std::size_t)
{
  auto window = m_shared_windows.find(ptr);
  if (window != m_shared_windows.end()) {
    MPI_Win_free(window.second);
  } else {
    UMPIRE_ERROR(umpire::unknown_pointer_error, fmt::fmt(
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
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
