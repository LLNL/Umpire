//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/NoOpMemoryResource.hpp"

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
  m_ptr = m_allocator.allocate(42*sizeof(int)); 
  m_count = 0;
}

NoOpMemoryResource::~NoOpMemoryResource()
{
  m_allocator.deallocate(m_ptr);
  m_count = 0;
}

void* NoOpMemoryResource::allocate(std::size_t bytes)
{
  UMPIRE_USE_VAR(bytes);
  m_count++; 
  return ((int*)m_ptr+m_count);
}

void NoOpMemoryResource::deallocate(void* ptr)
{
  UMPIRE_USE_VAR(ptr);
  m_count--;
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
  if(p == Platform::host)
    return true;
  else
    return false;
}

Platform NoOpMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
