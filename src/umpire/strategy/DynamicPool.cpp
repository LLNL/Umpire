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
#include "umpire/strategy/DynamicPool.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

DynamicPool::DynamicPool(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::size_t min_initial_alloc_size,
    const std::size_t min_alloc_size) noexcept :
  AllocationStrategy(name, id),
  dpa(nullptr),
  m_allocator(allocator.getAllocationStrategy())
{
  dpa = new DynamicSizePool<>(m_allocator, min_initial_alloc_size, min_alloc_size);
}

void*
DynamicPool::allocate(size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");
  void* ptr = dpa->allocate(bytes);
  return ptr;
}

void
DynamicPool::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  dpa->deallocate(ptr);
}

long
DynamicPool::getCurrentSize() noexcept
{
  return 0;
}

long
DynamicPool::getHighWatermark() noexcept
{
  return 0;
}

long
DynamicPool::getActualSize() noexcept
{
  long totalSize = dpa->totalSize();
  UMPIRE_LOG(Debug, "() returning " << totalSize);
  return totalSize;
}

Platform
DynamicPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void
DynamicPool::coalesce() noexcept
{
  dpa->coalesce();
}

} // end of namespace strategy
} // end of namespace umpire
