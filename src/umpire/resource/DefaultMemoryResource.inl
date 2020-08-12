//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DefaultMemoryResource_INL
#define UMPIRE_DefaultMemoryResource_INL

#include <memory>
#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

template <typename _allocator>
DefaultMemoryResource<_allocator>::DefaultMemoryResource(
    Platform platform, const std::string& name, int id,
    MemoryResourceTraits traits)
    : MemoryResource(name, id, traits), m_allocator(), m_platform(platform)
{
}

template <typename _allocator>
DefaultMemoryResource<_allocator>::DefaultMemoryResource(
    Platform platform, const std::string& name, int id,
    MemoryResourceTraits traits, _allocator alloc)
    : MemoryResource(name, id, traits), m_allocator(alloc), m_platform(platform)
{
}

template <typename _allocator>
void* DefaultMemoryResource<_allocator>::allocate(std::size_t bytes)
{
  void* ptr = m_allocator.allocate(bytes);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr),
                          "size", bytes, "event", "allocate");

  return ptr;
}

template <typename _allocator>
void DefaultMemoryResource<_allocator>::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr),
                          "size", 0x0, "event", "deallocate");

  m_allocator.deallocate(ptr);
}

template <typename _allocator>
std::size_t DefaultMemoryResource<_allocator>::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

template <typename _allocator>
std::size_t DefaultMemoryResource<_allocator>::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << 0);
  return 0;
}

template <typename _allocator>
Platform DefaultMemoryResource<_allocator>::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_DefaultMemoryResource_INL
