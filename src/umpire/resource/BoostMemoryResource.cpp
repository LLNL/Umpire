//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/BoostMemoryResource.hpp"
#include "umpire/resource/BoostMemoryResourceImpl.hpp"
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

BoostMemoryResource::BoostMemoryResource(Platform platform,
                                       const std::string& name, int id,
                                       MemoryResourceTraits traits)
    :
      MemoryResource{name, id, traits},
      m_platform{platform},
      pimpl{ new impl{ name, traits.size } }
{
}

BoostMemoryResource::~BoostMemoryResource()
{
}

void* BoostMemoryResource::allocate(std::size_t UMPIRE_UNUSED_ARG(bytes))
{
  UMPIRE_ERROR("Shared memory allocation without name is not supported");
}

void* BoostMemoryResource::allocate(const std::string& name, std::size_t bytes)
{
  return pimpl->allocate(name, bytes);
}

void BoostMemoryResource::deallocate(void* ptr)
{
  return pimpl->deallocate(ptr);
}

Platform BoostMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
