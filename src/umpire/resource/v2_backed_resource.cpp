//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/v2_backed_resource.hpp"

#include <utility>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

v2_backed_resource::v2_backed_resource(const std::string& name, int id, MemoryResourceTraits traits,
                                        Platform platform, std::unique_ptr<umpire::memory> v2_memory,
                                        AccessibilityFn is_accessible_from)
    : MemoryResource(name, id, traits),
      m_platform(platform),
      m_v2_memory(std::move(v2_memory)),
      m_is_accessible_from(std::move(is_accessible_from))
{
}

void* v2_backed_resource::allocate(std::size_t bytes)
{
  void* ptr = m_v2_memory->allocate(bytes);

  UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);

  return ptr;
}

void v2_backed_resource::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  m_v2_memory->deallocate(ptr);
}

Platform v2_backed_resource::getPlatform() noexcept
{
  return m_platform;
}

bool v2_backed_resource::isAccessibleFrom(Platform p) noexcept
{
  if (!m_is_accessible_from) {
    return false;
  }

  return m_is_accessible_from(p);
}

} // namespace resource
} // namespace umpire
