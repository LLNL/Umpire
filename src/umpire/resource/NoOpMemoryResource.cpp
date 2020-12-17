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

static const std::size_t Max_Allocations{100000};

NoOpMemoryResource::NoOpMemoryResource(Platform platform,
                                       const std::string& name, int id,
                                       MemoryResourceTraits traits)
    : MemoryResource{name, id, traits}, m_platform{platform}
{
  //Do not change value of Max_Allocations unless it is also
  //updated in NoOp benchmark.
  m_ptr = static_cast<char*>(m_allocator.allocate(Max_Allocations*sizeof(char)));
  m_count = 0;
}

NoOpMemoryResource::~NoOpMemoryResource()
{
  m_allocator.deallocate((void*)m_ptr);
  m_count = 0;
}

void* NoOpMemoryResource::allocate(std::size_t bytes)
{
  UMPIRE_USE_VAR(bytes);
  m_ptr+=m_count++;
  return (void*)m_ptr;
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
