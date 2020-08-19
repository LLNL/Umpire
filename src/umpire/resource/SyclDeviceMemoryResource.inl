//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SyclDeviceMemoryResource_INL
#define UMPIRE_SyclDeviceMemoryResource_INL

#include <memory>
#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/resource/SyclDeviceMemoryResource.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

template <typename _allocator>
SyclDeviceMemoryResource<_allocator>::SyclDeviceMemoryResource(
    Platform platform, const std::string& name, int id,
    MemoryResourceTraits traits)
    : MemoryResource(name, id, traits), m_allocator(), m_platform(platform)
{
}

template <typename _allocator>
void* SyclDeviceMemoryResource<_allocator>::allocate(std::size_t bytes)
{
  cl::sycl::queue sycl_queue(m_traits.queue);

  void* ptr = m_allocator.allocate(bytes, sycl_queue);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr),
                          "size", bytes, "event", "allocate");

  return ptr;
}

template <typename _allocator>
void SyclDeviceMemoryResource<_allocator>::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr),
                          "size", 0x0, "event", "deallocate");

  cl::sycl::queue sycl_queue(m_traits.queue);

  m_allocator.deallocate(ptr, sycl_queue);
}

template <typename _allocator>
std::size_t SyclDeviceMemoryResource<_allocator>::getCurrentSize() const
    noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

template <typename _allocator>
std::size_t SyclDeviceMemoryResource<_allocator>::getHighWatermark() const
    noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

template <typename _allocator>
Platform SyclDeviceMemoryResource<_allocator>::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_SyclDeviceMemoryResource_INL
