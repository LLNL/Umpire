//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef __Host_Shared_Memory_Resource_HPP
#define __Host_Shared_Memory_Resource_HPP

#include <memory>
#include <string>

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Platform.hpp"

namespace umpire {
namespace resource {

class HostSharedMemoryResource : public MemoryResource {
  public:
    HostSharedMemoryResource(Platform platform, const std::string& name, int id,
                      MemoryResourceTraits traits);

    ~HostSharedMemoryResource();

    void* allocate(std::size_t bytes);

    void* allocate(const std::string& name, std::size_t bytes);

    void deallocate(void* ptr);

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    bool isAccessibleFrom(Platform p) noexcept;
    Platform getPlatform() noexcept;

    void* find_pointer_from_name(std::string name);
  protected:
    Platform m_platform;
  private:
    class impl;
    std::unique_ptr<impl> pimpl;
};

} // end of namespace resource
} // end of namespace umpire
#endif // __Host_Shared_Memory_Resource_HPP
