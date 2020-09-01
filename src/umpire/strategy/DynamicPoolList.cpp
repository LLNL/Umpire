//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/DynamicPoolList.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/Replay.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

DynamicPoolList::DynamicPoolList(
    const std::string& name, int id, Allocator allocator,
    const std::size_t first_minimum_pool_allocation_size,
    const std::size_t next_minimum_pool_allocation_size,
    const std::size_t alignment, CoalesceHeuristic should_coalesce) noexcept
    : AllocationStrategy{name, id},
      m_allocator{allocator.getAllocationStrategy()},
      dpa{m_allocator, first_minimum_pool_allocation_size,
          next_minimum_pool_allocation_size, alignment},
      m_should_coalesce{should_coalesce}
{
}

void* DynamicPoolList::allocate(size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");

  void* ptr = dpa.allocate(bytes);
  return ptr;
}

void DynamicPoolList::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  dpa.deallocate(ptr);

  if (m_should_coalesce(*this)) {
    UMPIRE_LOG(Debug,
               "Heuristic returned true, "
               "performing coalesce operation for "
                   << this << "\n");
    dpa.coalesce();
  }
}

void DynamicPoolList::release()
{
  dpa.release();
}

std::size_t DynamicPoolList::getActualSize() const noexcept
{
  std::size_t ActualSize = dpa.getActualSize();
  UMPIRE_LOG(Debug, "() returning " << ActualSize);
  return ActualSize;
}

std::size_t DynamicPoolList::getReleasableSize() const noexcept
{
  std::size_t SparseBlockSize = dpa.getReleasableSize();
  UMPIRE_LOG(Debug, "() returning " << SparseBlockSize);
  return SparseBlockSize;
}

std::size_t DynamicPoolList::getBlocksInPool() const noexcept
{
  std::size_t BlocksInPool = dpa.getBlocksInPool();
  UMPIRE_LOG(Debug, "() returning " << BlocksInPool);
  return BlocksInPool;
}

std::size_t DynamicPoolList::getLargestAvailableBlock() const noexcept
{
  std::size_t LargestAvailableBlock = dpa.getLargestAvailableBlock();
  UMPIRE_LOG(Debug, "() returning " << LargestAvailableBlock);
  return LargestAvailableBlock;
}

Platform DynamicPoolList::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits DynamicPoolList::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

void DynamicPoolList::coalesce() noexcept
{
  UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \""
                << getName() << "\" }");
  dpa.coalesce();
}

DynamicPoolList::CoalesceHeuristic DynamicPoolList::percent_releasable(
    int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR("Invalid percentage of "
                 << percentage
                 << ", percentage must be an integer between 0 and 100");
  }

  if (percentage == 0) {
    return
        [=](const DynamicPoolList& UMPIRE_UNUSED_ARG(pool)) { return false; };
  } else if (percentage == 100) {
    return [=](const strategy::DynamicPoolList& pool) {
      return (pool.getActualSize() == pool.getReleasableSize());
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);

    return [=](const strategy::DynamicPoolList& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold =
          static_cast<std::size_t>(f * pool.getActualSize());
      return (pool.getReleasableSize() >= threshold);
    };
  }
}

} // end of namespace strategy
} // end of namespace umpire
