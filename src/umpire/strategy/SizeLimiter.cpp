//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/SizeLimiter.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {
namespace strategy {

SizeLimiter::SizeLimiter(const std::string& name, int id, Allocator allocator,
                         std::size_t size_limit)
    : AllocationStrategy(name, id),
      m_allocator(allocator.getAllocationStrategy()),
      m_size_limit(size_limit),
      m_total_size(0)
{
}

void* SizeLimiter::allocate(std::size_t bytes)
{
  m_total_size += bytes;

  if (m_total_size > m_size_limit) {
    m_total_size -= bytes;
    UMPIRE_ERROR("Size limit exceeded.");
  }

  return m_allocator->allocate(bytes);
}

void SizeLimiter::deallocate(void* ptr)
{
  m_total_size -= ResourceManager::getInstance().getSize(ptr);
  m_allocator->deallocate(ptr);
}

Platform SizeLimiter::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits SizeLimiter::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // end of namespace strategy
} // end of namespace umpire
