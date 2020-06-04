//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/SyclDeviceMemoryResource.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

#include <memory>
#include <sstream>

namespace umpire {
namespace resource {

SyclDeviceMemoryResource::SyclDeviceMemoryResource(
    Platform platform,
    const std::string& name,
    int id,
    MemoryResourceTraits traits) :
  MemoryResource(name, id, traits),
  m_allocator{},
  m_platform(platform)
{
}

void* SyclDeviceMemoryResource::allocate(std::size_t bytes)
{
  cl::sycl::device sycl_device(m_traits.deviceID);
  cl::sycl::queue sycl_queue(sycl_device);

  void* ptr = m_allocator.allocate(bytes, sycl_queue);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", bytes, "event", "allocate");

  return ptr;
}

void SyclDeviceMemoryResource::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

  cl::sycl::device sycl_device(m_traits.deviceID);
  cl::sycl::queue sycl_queue(sycl_device);

  m_allocator.deallocate(ptr, sycl_queue);
}

std::size_t SyclDeviceMemoryResource::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

std::size_t SyclDeviceMemoryResource::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

Platform SyclDeviceMemoryResource::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
