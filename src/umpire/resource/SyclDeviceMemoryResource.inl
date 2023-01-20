//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
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
SyclDeviceMemoryResource<_allocator>::SyclDeviceMemoryResource(Platform platform, const std::string& name, int id,
                                                               MemoryResourceTraits traits)
    : MemoryResource(name, id, traits), m_allocator(), m_platform(platform)
{
}

template <typename _allocator>
void* SyclDeviceMemoryResource<_allocator>::allocate(std::size_t bytes)
{
  void* ptr = m_allocator.allocate(bytes, *(m_traits.queue));

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);

  return ptr;
}

template <typename _allocator>
void SyclDeviceMemoryResource<_allocator>::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  m_allocator.deallocate(ptr, *(m_traits.queue));
}

template <typename _allocator>
std::size_t SyclDeviceMemoryResource<_allocator>::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

template <typename _allocator>
std::size_t SyclDeviceMemoryResource<_allocator>::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

template <typename _allocator>
bool SyclDeviceMemoryResource<_allocator>::isAccessibleFrom(Platform p) noexcept
{
  return m_allocator.isAccessible(p);
}

template <typename _allocator>
Platform SyclDeviceMemoryResource<_allocator>::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_SyclDeviceMemoryResource_INL
