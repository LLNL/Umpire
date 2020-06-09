//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/MpiSharedMemoryResource.hpp"

#include "umpire/util/Macros.hpp"

#include "mpi.h"

#include <string>

namespace umpire {
namespace resource {

MpiSharedMemoryResource::MpiSharedMemoryResource(
    Platform platform,
    const std::string& name,
    int id,
    MemoryResourceTraits traits) :
    SharedMemoryResource(name, id, traits)
  , m_platform{platform}
{
}

void MpiSharedMemoryResource::initialize()
{
  if (! m_initialized ) {
    char nodename[MPI_MAX_PROCESSOR_NAME];
    int nodestringlen;

    MPI_Get_processor_name(nodename, &nodestringlen);
    m_nodename = nodename;

    int rank;
    MPI_Comm_rank(m_allcomm, &rank);
    MPI_Comm_split_type(m_allcomm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &m_nodecomm);

    MPI_Comm_rank(m_nodecomm, &m_noderank);
    m_initialized = true;
  }
}

void* MpiSharedMemoryResource::allocate(std::size_t bytes)
{
  initialize();
  auto localsize = is_foreman() ? bytes : 0; // Foreman is the only one to actually allocate any memory
  void* ptr;
  MPI_Win window;

  MPI_Win_allocate_shared(localsize, 1, MPI_INFO_NULL, m_nodecomm, &ptr, &window);

  // TODO: Error Checking

  // need to get local pointer valid for ptr on foreman rank

  if ( !is_foreman() ) {
    MPI_Aint lsize;
    int disp_unit;

    //MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr);
    MPI_Win_shared_query(window, m_foremanrank, &lsize, &disp_unit, &ptr);
  }

  // TODO: Need to do error checking here

  m_winmap[ptr] = window;

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", bytes, "event", "allocate");

  return ptr;
}

void* MpiSharedMemoryResource::allocate(std::string name, std::size_t bytes)
{
  void* ptr = allocate(bytes);
  m_name_to_allocation[name] = ptr;
  m_allocation_to_name[ptr] = name;

  return ptr;
}

void* MpiSharedMemoryResource::get_allocation_by_name(std::string name)
{
  return m_name_to_allocation[name];
}

void MpiSharedMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

  auto window = m_winmap[ptr];

  m_name_to_allocation.erase(m_allocation_to_name[ptr]);
  m_allocation_to_name.erase(ptr);

  MPI_Win_free(&window);  // Frees the window.  Not really sure how the shared memory actually gets freed
}

std::size_t MpiSharedMemoryResource::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t MpiSharedMemoryResource::getHighWatermark() const noexcept
{
  return 0;
}

Platform MpiSharedMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

bool MpiSharedMemoryResource::is_foreman() noexcept
{
  return m_noderank == m_foremanrank;
}

void MpiSharedMemoryResource::synchronize() noexcept
{
  MPI_Barrier(m_allcomm);
}

void MpiSharedMemoryResource::set_foreman(int id) noexcept
{
  m_foremanrank = id;
}

} // end of namespace resource
} // end of namespace umpire