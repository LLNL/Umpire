//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/resource/HostSharedMemoryResourceImpl.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/Macros.hpp"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

namespace umpire {
namespace resource {

HostSharedMemoryResource::HostSharedMemoryResource(Platform platform,
                                       const std::string& name, int id,
                                       MemoryResourceTraits traits)
    :
      MemoryResource{name, id, traits},
      m_platform{platform},
      pimpl{ new impl{ name, traits.size } }
{
}

HostSharedMemoryResource::~HostSharedMemoryResource()
{
}

void* HostSharedMemoryResource::allocate(std::size_t UMPIRE_UNUSED_ARG(bytes))
{
  UMPIRE_ERROR("Shared memory allocation without name is not supported");
}

void* HostSharedMemoryResource::allocate(const std::string& name, std::size_t bytes)
{
  return pimpl->allocate(name, bytes);
}

void HostSharedMemoryResource::deallocate(void* ptr)
{
  return pimpl->deallocate(ptr);
}

Platform HostSharedMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

std::size_t HostSharedMemoryResource::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t HostSharedMemoryResource::getHighWatermark() const noexcept
{
  return 0;
}

void* HostSharedMemoryResource::find_pointer_from_name(std::string name)
{
  return pimpl->find_pointer_from_name(name);
}
} // end of namespace resource
} // end of namespace umpire
