//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_FileMemoryResource_HPP
#define UMPIRE_FileMemoryResource_HPP

#include "umpire/resource/MemoryResource.hpp"

#include "umpire/util/Platform.hpp"
#include "umpire/util/MemoryMap.hpp"

#include <utility>

namespace umpire {
namespace resource {

class FileMemoryResource :
  public MemoryResource
{
  public: 
    FileMemoryResource(Platform platform, const std::string& name, int id, MemoryResourceTraits traits);

    void* allocate(std::size_t bytes);
    void deallocate(void* ptr);

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;

  protected:
    Platform m_platform;

  private:
    util::MemoryMap<std::pair <const std::string, std::size_t>> m_size_map;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_FileMemoryResource_HPP
