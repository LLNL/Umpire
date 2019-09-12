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
#include "umpire/strategy/DynamicPoolList.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"

namespace umpire {
namespace strategy {

DynamicPoolList::DynamicPoolList(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::size_t min_initial_alloc_size,
    const std::size_t min_alloc_size,
    CoalesceHeuristic coalesce_heuristic) noexcept
  :
  AllocationStrategy(name, id),
  dpa(nullptr),
  m_allocator(allocator.getAllocationStrategy()),
  do_coalesce{coalesce_heuristic}
{
  dpa = new DynamicSizePool<>(m_allocator, min_initial_alloc_size, min_alloc_size);
}

void*
DynamicPoolList::allocate(size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");

  void* ptr = dpa->allocate(bytes);
  return ptr;
}

void
DynamicPoolList::deallocate(void* ptr)
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
DynamicPoolList::release()
{
  dpa->release();
}

std::size_t
DynamicPoolList::getCurrentSize() const noexcept
{
  long CurrentSize = dpa->getCurrentSize();
  UMPIRE_LOG(Debug, "() returning " << CurrentSize);
  return CurrentSize;
}

std::size_t
DynamicPoolList::getActualSize() const noexcept
{
  long ActualSize = dpa->getActualSize();
  UMPIRE_LOG(Debug, "() returning " << ActualSize);
  return ActualSize;
}

std::size_t
DynamicPoolList::getHighWatermark() const noexcept
{
  long HighWatermark = dpa->getHighWatermark();
  UMPIRE_LOG(Debug, "() returning " << HighWatermark);
  return HighWatermark;
}

std::size_t
DynamicPoolList::getReleasableSize() const noexcept
{
  long SparseBlockSize = dpa->getReleasableSize();
  UMPIRE_LOG(Debug, "() returning " << SparseBlockSize);
  return SparseBlockSize;
}

std::size_t
DynamicPoolList::getBlocksInPool() const noexcept
{
  long BlocksInPool = dpa->getBlocksInPool();
  UMPIRE_LOG(Debug, "() returning " << BlocksInPool);
  return BlocksInPool;
}

Platform
DynamicPoolList::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void
DynamicPoolList::coalesce() noexcept
{
  UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \"" << getName() << "\" }");
  dpa->coalesce();
}

} // end of namespace strategy
} // end of namespace umpire
