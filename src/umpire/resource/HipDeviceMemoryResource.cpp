//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
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

HipDeviceMemoryResource::HipDeviceMemoryResource(Platform platform, const std::string& name, int id,
                                                 MemoryResourceTraits traits)
    : MemoryResource(name, id, traits), m_allocator{traits.granularity}, m_platform(platform)
{
}

void* HipDeviceMemoryResource::allocate(std::size_t bytes)
{
  int old_device;
  hipError_t err = hipGetDevice(&old_device);
  if (err != hipSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("hipGetDevice failed with error: {}", hipGetErrorString(err)));
  }
  if (old_device != m_traits.id) {
    err = hipSetDevice(m_traits.id);
    if (err != hipSuccess) {
      UMPIRE_ERROR(runtime_error, fmt::format("hipSetDevice( device = {} ) failed with error: {}", m_traits.id,
                                              hipGetErrorString(err)));
    }
  }

  void* ptr = m_allocator.allocate(bytes);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);

  if (old_device != m_traits.id) {
    err = hipSetDevice(old_device);
    if (err != hipSuccess) {
      UMPIRE_ERROR(runtime_error, fmt::format("hipSetDevice( device = {} ) failed with error: {}", old_device,
                                              hipGetErrorString(err)));
    }
  }
  return ptr;
}

void HipDeviceMemoryResource::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  int old_device;
  hipError_t err = hipGetDevice(&old_device);
  if (err != hipSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("hipGetDevice failed with error: {}", hipGetErrorString(err)));
  }
  if (old_device != m_traits.id) {
    err = hipSetDevice(m_traits.id);
    if (err != hipSuccess) {
      UMPIRE_ERROR(runtime_error, fmt::format("hipSetDevice( device = {} ) failed with error: {}", m_traits.id,
                                              hipGetErrorString(err)));
    }
  }

  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  m_allocator.deallocate(ptr);
  if (old_device != m_traits.id) {
    err = hipSetDevice(old_device);
    if (err != hipSuccess) {
      UMPIRE_ERROR(runtime_error, fmt::format("hipSetDevice( device = {} ) failed with error: {}", old_device,
                                              hipGetErrorString(err)));
    }
  }
}

bool HipDeviceMemoryResource::isAccessibleFrom(Platform p) noexcept
{
  if (p == Platform::hip || p == Platform::host)
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
