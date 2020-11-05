//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_NullMemoryResource_HPP
#define UMPIRE_NullMemoryResource_HPP

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/MemoryMap.hpp"
#include "camp/resource/platform.hpp"

namespace umpire {
namespace resource {

class NullMemoryResource : public MemoryResource {
 public:
  NullMemoryResource(camp::resources::Platform platform, const std::string& name, int id,
                     MemoryResourceTraits traits);

  void* allocate(std::size_t bytes);
  void deallocate(void* ptr);

  std::size_t getCurrentSize() const noexcept;
  std::size_t getHighWatermark() const noexcept;

  camp::resources::Platform getPlatform() noexcept;

 protected:
  camp::resources::Platform m_platform;

 private:
  util::MemoryMap<size_t> m_size_map;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_NullMemoryResource_HPP
