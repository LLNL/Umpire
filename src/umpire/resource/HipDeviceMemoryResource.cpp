//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipDeviceMemoryResource.hpp"

#include <memory>
#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

HipDeviceMemoryResource::HipDeviceMemoryResource(Platform platform,
                                                   const std::string& name,
                                                   int id,
                                                   MemoryResourceTraits traits)
    : MemoryResource(name, id, traits), m_allocator{}, m_platform(platform)
{
}

void* HipDeviceMemoryResource::allocate(std::size_t bytes)
{
  int old_device;
  hipGetDevice(&old_device);
  if (old_device != m_traits.id)
    hipSetDevice(m_traits.id);

  void* ptr = m_allocator.allocate(bytes);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);

  if (old_device != m_traits.id)
    hipSetDevice(old_device);
  return ptr;
}

void HipDeviceMemoryResource::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  int old_device;
  hipGetDevice(&old_device);
  if (old_device != m_traits.id)
    hipSetDevice(m_traits.id);

  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  m_allocator.deallocate(ptr);
  if (old_device != m_traits.id)
    hipSetDevice(old_device);
}

std::size_t HipDeviceMemoryResource::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

std::size_t HipDeviceMemoryResource::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

bool HipDeviceMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if(p == Platform::hip || p == Platform::host)
    return true;
  else
    return false;
}

Platform HipDeviceMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
