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
    int _id,
    Allocator allocator,
    const std::size_t min_initial_alloc_size,
    const std::size_t min_alloc_size,
    Coalesce_Heuristic h_fun) noexcept
  :
  AllocationStrategy(name, _id),
  dpa(nullptr),
  m_allocator(allocator.getAllocationStrategy()),
  do_coalesce{h_fun}
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

  if ( do_coalesce(*this) ) {
    UMPIRE_LOG(Debug, "Heuristic returned true, "
        "performing coalesce operation for " << this << "\n");
    dpa->coalesce();
  }
}

void
DynamicPool::release()
{
  dpa->release();
}

long
DynamicPool::getCurrentSize() noexcept
{
  long CurrentSize = dpa->getCurrentSize();
  UMPIRE_LOG(Debug, "() returning " << CurrentSize);
  return CurrentSize;
}

long
DynamicPool::getActualSize() noexcept
{
  long ActualSize = dpa->getActualSize();
  UMPIRE_LOG(Debug, "() returning " << ActualSize);
  return ActualSize;
}

long
DynamicPool::getHighWatermark() noexcept
{
  long HighWatermark = dpa->getHighWatermark();
  UMPIRE_LOG(Debug, "() returning " << HighWatermark);
  return HighWatermark;
}

long
DynamicPool::getReleaseableSize() noexcept
{
  long SparseBlockSize = dpa->getReleaseableSize();
  UMPIRE_LOG(Debug, "() returning " << SparseBlockSize);
  return SparseBlockSize;
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
