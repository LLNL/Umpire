//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/NamedAllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

NamedAllocationStrategy::NamedAllocationStrategy(
    const std::string& name,
    int id,
    Allocator allocator) :
  AllocationStrategy(name, id),
  m_allocator(allocator.getAllocationStrategy())
{
}

void* 
NamedAllocationStrategy::allocate(std::size_t bytes)
{
  return m_allocator->allocate(bytes);
}

void 
NamedAllocationStrategy::deallocate(void* ptr)
{
  return m_allocator->deallocate(ptr);
}

std::size_t 
NamedAllocationStrategy::getCurrentSize() const noexcept
{
  return 0;
}

std::size_t 
NamedAllocationStrategy::getHighWatermark() const noexcept
{
  return 0;
}

Platform 
NamedAllocationStrategy::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
