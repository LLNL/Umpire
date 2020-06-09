//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MpiSharedMemoryResource_HPP
#define UMPIRE_MpiSharedMemoryResource_HPP

#include "umpire/resource/SharedMemoryResource.hpp"

#include "umpire/util/Platform.hpp"

#include "mpi.h"

#include <string>
#include <unordered_map>

namespace umpire {
namespace resource {

class MpiSharedMemoryResource :
  public SharedMemoryResource
{
  public:
    MpiSharedMemoryResource(Platform platform, const std::string& name, int id, MemoryResourceTraits traits);

    void* allocate(std::size_t bytes);
    void* allocate(std::string name, std::size_t bytes);
    virtual void* get_allocation_by_name(std::string allocation_name);
    void deallocate(void* ptr);

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;
    void set_foreman(int id) noexcept;
    bool is_foreman() noexcept;
    void synchronize() noexcept;

  private:
    bool m_initialized{false};
    Platform m_platform;
    MPI_Comm m_allcomm{MPI_COMM_WORLD};
    MPI_Comm m_nodecomm;
    std::string m_nodename;
    int m_foremanrank{0};
    int m_noderank;
    std::unordered_map<void*, MPI_Win> m_winmap;

    std::unordered_map<std::string, void*> m_name_to_allocation;
    std::unordered_map<void*, std::string> m_allocation_to_name;

    void initialize() noexcept;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_MpiSharedMemoryResource_HPP
