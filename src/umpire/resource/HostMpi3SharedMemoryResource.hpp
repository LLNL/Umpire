//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef __Host_Shared_Memory_Resource_HPP
#define __Host_Shared_Memory_Resource_HPP

#include <memory>
#include <string>
#include <map>

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

class HostMpi3SharedMemoryResource : public MemoryResource {
 public:
  HostMpi3SharedMemoryResource(Platform platform, const std::string& name, int id, MemoryResourceTraits traits);

  ~HostMpi3SharedMemoryResource();

  void* allocate(std::size_t bytes) override;

  void* allocate_named(const std::string& name, std::size_t bytes) override;

  void deallocate(void* ptr, std::size_t size) override;

  std::size_t getActualSize() const noexcept override;

  bool isAccessibleFrom(Platform p) noexcept override;
  Platform getPlatform() noexcept override;
 protected:
  Platform m_platform;

 private:
  MPI_comm m_shared_comm;
  int m_local_rank;
  std::map<void*, MPI_Win> m_shared_windows;
};

} // end of namespace resource
} // end of namespace umpire
#endif // __Host_Shared_Memory_Resource_HPP
