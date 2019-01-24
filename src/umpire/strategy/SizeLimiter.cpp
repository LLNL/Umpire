//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/SizeLimiter.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {
namespace strategy {

SizeLimiter::SizeLimiter(
    const std::string& name,
    int id,
    Allocator allocator,
    size_t size_limit) :
  AllocationStrategy(name, id),
  m_allocator(allocator.getAllocationStrategy()),
  m_size_limit(size_limit),
  m_total_size(0)
{
}

void* SizeLimiter::allocate(size_t bytes)
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

long SizeLimiter::getCurrentSize() const noexcept
{
  return 0;
}

long SizeLimiter::getHighWatermark() const noexcept
{
  return 0;
}

Platform SizeLimiter::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
