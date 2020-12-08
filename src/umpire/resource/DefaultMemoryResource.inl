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

  return ptr;
}

template <typename _allocator>
void DefaultMemoryResource<_allocator>::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  m_allocator.deallocate(ptr);
}

template <typename _allocator>
bool DefaultMemoryResource<_allocator>::isAccessibleFrom(Platform p) noexcept
{
  return m_allocator.isAccessible(p);
}

template <typename _allocator>
Platform DefaultMemoryResource<_allocator>::getPlatform() noexcept
{
  return m_platform;
}

} // end of namespace resource
} // end of namespace umpire
#endif // UMPIRE_DefaultMemoryResource_INL
