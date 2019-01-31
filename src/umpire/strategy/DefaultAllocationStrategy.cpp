//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include "umpire/strategy/DefaultAllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

DefaultAllocationStrategy::DefaultAllocationStrategy(std::shared_ptr<AllocationStrategy> allocator) :
  AllocationStrategy("DEFAULT"),
  m_allocator(allocator)
{
}

void* 
DefaultAllocationStrategy::allocate(size_t bytes)
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
