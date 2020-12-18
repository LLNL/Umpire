//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/NoOpMemoryResource.hpp"
#include <cstdint>
#include <stdlib.h>
#include <string.h>
#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {


NoOpMemoryResource::NoOpMemoryResource(Platform platform,
                                       const std::string& name, int id,
                                       MemoryResourceTraits traits)
    : MemoryResource{name, id, traits}, m_platform{platform}
{
  m_count = (UINT64_C(1)<<48);
}

NoOpMemoryResource::~NoOpMemoryResource()
{
  m_count = (UINT64_C(1)<<48);
}

void* NoOpMemoryResource::allocate(std::size_t bytes)
{
  void* ptr = (void*)m_count;
  m_count += bytes;
  return ptr;
}

void NoOpMemoryResource::deallocate(void* ptr)
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
  if(p != Platform::undefined)
    UMPIRE_LOG(Debug, "NullMemoryResource: platform is not accessible");
  return false;
}

Platform NoOpMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
