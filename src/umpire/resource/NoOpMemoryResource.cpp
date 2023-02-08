//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/NoOpMemoryResource.hpp"

#include <stdlib.h>
#include <string.h>

#include <cstdint>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

NoOpMemoryResource::NoOpMemoryResource(Platform platform, const std::string& name, int id, MemoryResourceTraits traits)
    : MemoryResource{name, id, traits}, m_platform{platform}
{
}

NoOpMemoryResource::~NoOpMemoryResource()
{
}

void* NoOpMemoryResource::allocate(std::size_t bytes)
{
  void* ptr = (void*)m_count;
  m_count += bytes;
  return ptr;
}

void NoOpMemoryResource::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  UMPIRE_USE_VAR(ptr);
}

std::size_t NoOpMemoryResource::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t NoOpMemoryResource::getHighWatermark() const noexcept
{
  return 0;
}

bool NoOpMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if (p != Platform::undefined)
    UMPIRE_LOG(Debug, "NullMemoryResource: platform is not accessible");
  return false;
}

Platform NoOpMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
