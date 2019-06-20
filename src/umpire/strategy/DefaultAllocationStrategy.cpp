//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/DefaultAllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

DefaultAllocationStrategy::DefaultAllocationStrategy(strategy::AllocationStrategy* allocator) :
  AllocationStrategy("DEFAULT"),
  m_allocator(allocator)
{
}

void* 
DefaultAllocationStrategy::allocate(std::size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");
  return m_allocator->allocate(bytes);
}

void 
DefaultAllocationStrategy::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  return m_allocator->deallocate(ptr);
}

long 
DefaultAllocationStrategy::getCurrentSize() const
{
  return m_allocator->getCurrentSize();
}

long 
DefaultAllocationStrategy::getHighWatermark() const
{
  return m_allocator->getHighWatermark();
}

Platform 
DefaultAllocationStrategy::getPlatform()
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
