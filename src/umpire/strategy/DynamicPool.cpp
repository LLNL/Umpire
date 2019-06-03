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
#include "umpire/strategy/DynamicPool.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"

namespace umpire {
namespace strategy {

DynamicPool::DynamicPool(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::size_t min_initial_alloc_size,
    const std::size_t min_alloc_size,
    Coalesce_Heuristic coalesce_heuristic) noexcept
  :
  AllocationStrategy(name, id),
  dpa(m_allocator, min_initial_alloc_size, min_alloc_size),
  m_allocator(allocator.getAllocationStrategy()),
  do_coalesce{coalesce_heuristic}
{
}

void*
DynamicPool::allocate(size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");

  void* ptr = dpa.allocate(bytes);
  return ptr;
}

void
DynamicPool::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  dpa.deallocate(ptr);

  if ( do_coalesce(*this) ) {
    UMPIRE_LOG(Debug, "Heuristic returned true, "
        "performing coalesce operation for " << this << "\n");
    coalesce();
  }
}

void
DynamicPool::release()
{
  dpa.release();
}

long
DynamicPool::getCurrentSize() const noexcept
{
  long CurrentSize = dpa.getCurrentSize();
  UMPIRE_LOG(Debug, "() returning " << CurrentSize);
  return CurrentSize;
}

long
DynamicPool::getActualSize() const noexcept
{
  long ActualSize = dpa.getActualSize();
  UMPIRE_LOG(Debug, "() returning " << ActualSize);
  return ActualSize;
}

long
DynamicPool::getHighWatermark() const noexcept
{
  long HighWatermark = dpa.getHighWatermark();
  UMPIRE_LOG(Debug, "() returning " << HighWatermark);
  return HighWatermark;
}

long
DynamicPool::getReleasableSize() const noexcept
{
  long SparseBlockSize = dpa.getReleasableSize();
  UMPIRE_LOG(Debug, "() returning " << SparseBlockSize);
  return SparseBlockSize;
}

long
DynamicPool::getBlocksInPool() const noexcept
{
  long BlocksInPool = dpa.getBlocksInPool();
  UMPIRE_LOG(Debug, "() returning " << BlocksInPool);
  return BlocksInPool;
}

Platform
DynamicPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void
DynamicPool::coalesce() noexcept
{
  UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \"" << getName() << "\" }");
  dpa.coalesce();
}

} // end of namespace strategy
} // end of namespace umpire
