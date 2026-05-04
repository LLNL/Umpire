//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Host_Mpi3_Shared_Memory_Resource_HPP
#define UMPIRE_Host_Mpi3_Shared_Memory_Resource_HPP

#include <map>
#include <memory>
#include <string>

#include "mpi.h"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

class HostMpi3SharedMemoryResource : public MemoryResource {
 public:
  HostMpi3SharedMemoryResource(const std::string& name, int id, MemoryResourceTraits traits);

  ~HostMpi3SharedMemoryResource();

  void* allocate(std::size_t bytes) override;

  void deallocate(void* ptr, std::size_t size) override;

  bool isAccessibleFrom(Platform p) noexcept override;

  Platform getPlatform() noexcept override;

  MPI_Comm getSharedCommunicator() const noexcept;

 private:
  static int free_comm(MPI_Comm comm, int keyval, void* attribute_val, void* extra_state);

  MPI_Comm m_shared_comm;
  int m_local_rank;
  std::map<void*, MPI_Win> m_shared_windows;
};

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_Host_Mpi3_Shared_Memory_Resource_HPP
