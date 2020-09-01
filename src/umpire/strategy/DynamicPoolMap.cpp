//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/DynamicPoolMap.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>

#include "umpire/Allocator.hpp"
#include "umpire/Replay.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/mixins/AlignedAllocation.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/backtrace.hpp"
#include "umpire/util/memory_sanitizers.hpp"

namespace umpire {
namespace strategy {

DynamicPoolMap::DynamicPoolMap(
    const std::string& name, int id, Allocator allocator,
    const std::size_t first_minimum_pool_allocation_size,
    const std::size_t next_minimum_pool_allocation_size,
    const std::size_t alignment, CoalesceHeuristic should_coalesce) noexcept
    : AllocationStrategy{name, id},
      mixins::AlignedAllocation{alignment, allocator.getAllocationStrategy()},
      m_should_coalesce{should_coalesce},
      m_first_minimum_pool_allocation_size{
          aligned_round_up(first_minimum_pool_allocation_size)},
      m_next_minimum_pool_allocation_size{
          aligned_round_up(next_minimum_pool_allocation_size)}
{
}

DynamicPoolMap::~DynamicPoolMap()
{
  // Get as many whole blocks as possible in the m_free_map
  mergeFreeBlocks();

  // Free any unused blocks
  for (auto& rec : m_free_map) {
    const std::size_t bytes{rec.first};
    void* addr;
    bool is_head;
    std::size_t whole_bytes;
    std::tie(addr, is_head, whole_bytes) = rec.second;
    // Deallocate if this is a whole block
    if (is_head && bytes == whole_bytes) {
      deallocateBlock(addr, bytes);
    }
  }

  if (m_used_map.size() == 0) {
    UMPIRE_ASSERT(m_actual_bytes == 0);
  }
}

void DynamicPoolMap::insertUsed(Pointer addr, std::size_t bytes, bool is_head,
                                std::size_t whole_bytes)
{
  m_used_map.insert(
      std::make_pair(addr, std::make_tuple(bytes, is_head, whole_bytes)));
}

void DynamicPoolMap::insertFree(Pointer addr, std::size_t bytes, bool is_head,
                                std::size_t whole_bytes)
{
  m_free_map.insert(
      std::make_pair(bytes, std::make_tuple(addr, is_head, whole_bytes)));
}

DynamicPoolMap::SizeMap::const_iterator DynamicPoolMap::findFreeBlock(
    std::size_t bytes) const
{
  SizeMap::const_iterator iter{m_free_map.upper_bound(bytes)};

  if (iter != m_free_map.begin()) {
    // Back up iterator
    --iter;
    const std::size_t test_bytes{iter->first};
    if (test_bytes < bytes) {
      // Too small, reset iterator to what upper_bound returned
      ++iter;
    }
  }

  return iter;
}

void* DynamicPoolMap::allocateBlock(std::size_t bytes)
{
  void* ptr{nullptr};
  try {
#if defined(UMPIRE_ENABLE_BACKTRACE)
    {
      umpire::util::backtrace bt;
      umpire::util::backtracer<>::get_backtrace(bt);
      UMPIRE_LOG(Info,
                 "actual_size: " << (m_actual_bytes + bytes)
                                 << " (prev: " << m_actual_bytes << ") "
                                 << umpire::util::backtracer<>::print(bt));
    }
#endif
    ptr = aligned_allocate(bytes);
  } catch (...) {
    UMPIRE_LOG(Error,
               "\n\tMemory exhausted at allocation resource. "
               "Attempting to give blocks back.\n\t"
                   << getFreeBlocks() << " Free Blocks, " << getInUseBlocks()
                   << " Used Blocks\n");
    mergeFreeBlocks();
    releaseFreeBlocks();
    UMPIRE_LOG(Error,
               "\n\tMemory exhausted at allocation resource.  "
               "\n\tRetrying allocation operation: "
                   << getFreeBlocks() << " Free Blocks, " << getInUseBlocks()
                   << " Used Blocks\n");
    try {
      ptr = aligned_allocate(bytes);
      UMPIRE_LOG(Error,
                 "\n\tMemory successfully recovered at resource.  Allocation "
                 "succeeded\n");
    } catch (...) {
      UMPIRE_LOG(Error,
                 "\n\tUnable to allocate from resource even after giving back "
                 "free blocks.\n"
                 "\tThrowing to let application know we have no more memory: "
                     << getFreeBlocks() << " Partially Free Blocks, "
                     << getInUseBlocks() << " Used Blocks\n");
      throw;
    }
  }

  UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, bytes);

  m_actual_bytes += bytes;

  return ptr;
}

void DynamicPoolMap::deallocateBlock(void* ptr, std::size_t size)
{
  UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, size);
  m_actual_bytes -= size;
  aligned_deallocate(ptr);
}

void* DynamicPoolMap::allocate(std::size_t bytes)
{
  bytes = aligned_round_up(bytes);
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");

  Pointer ptr{nullptr};

  // Check if the previous block is a match
  const SizeMap::const_iterator iter{findFreeBlock(bytes)};

  if (iter != m_free_map.end()) {
    // Found this acceptable address pair
    bool is_head;
    std::size_t whole_bytes;
    std::tie(ptr, is_head, whole_bytes) = iter->second;

    // Add used map
    insertUsed(ptr, bytes, is_head, whole_bytes);

    // Remove the entry from the free map
    const std::size_t free_size{iter->first};
    m_free_map.erase(iter);

    const std::size_t left_bytes{free_size - bytes};

    if (left_bytes > 0) {
      insertFree(static_cast<unsigned char*>(ptr) + bytes, left_bytes, false,
                 whole_bytes);
    }
  } else {
    const std::size_t min_block_size =
        (m_actual_bytes == 0) ? m_first_minimum_pool_allocation_size
                              : m_next_minimum_pool_allocation_size;

    const std::size_t alloc_bytes{std::max(bytes, min_block_size)};
    ptr = allocateBlock(alloc_bytes);

    UMPIRE_ASSERT("bytes too large" && bytes <= alloc_bytes);

    insertUsed(ptr, bytes, true, alloc_bytes);

    const std::size_t left_bytes{alloc_bytes - bytes};

    // Add free
    if (left_bytes > 0)
      insertFree(static_cast<unsigned char*>(ptr) + bytes, left_bytes, false,
                 alloc_bytes);
  }

  UMPIRE_UNPOISON_MEMORY_REGION(m_allocator, ptr, bytes);
  return ptr;
}

void DynamicPoolMap::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  UMPIRE_ASSERT(ptr);

  auto iter = m_used_map.find(ptr);

  if (iter->second) {
    // Fast way to check if key was found

    std::size_t bytes;
    bool is_head;
    std::size_t whole_bytes;
    std::tie(bytes, is_head, whole_bytes) = *iter->second;

    // Insert in free map
    insertFree(ptr, bytes, is_head, whole_bytes);

    // Remove from used map
    m_used_map.erase(iter);

    UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, bytes);
  } else {
    UMPIRE_ERROR("Cound not found ptr = " << ptr);
  }

  if (m_should_coalesce(*this)) {
    UMPIRE_LOG(Debug,
               this << " heuristic function returned true, calling coalesce()");
    do_coalesce();
  }
}

std::size_t DynamicPoolMap::getActualSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_actual_bytes);
  return m_actual_bytes;
}

std::size_t DynamicPoolMap::getFreeBlocks() const noexcept
{
  return m_free_map.size();
}

std::size_t DynamicPoolMap::getInUseBlocks() const noexcept
{
  return m_used_map.size();
}

std::size_t DynamicPoolMap::getBlocksInPool() const noexcept
{
  const std::size_t total_blocks{getFreeBlocks() + getInUseBlocks()};
  UMPIRE_LOG(Debug, "() returning " << total_blocks);
  return total_blocks;
}

std::size_t DynamicPoolMap::getLargestAvailableBlock() noexcept
{
  std::size_t largest_block{0};

  mergeFreeBlocks();

  for (auto& rec : m_free_map) {
    const std::size_t bytes{rec.first};
    if (bytes > largest_block)
      largest_block = bytes;
  }

  UMPIRE_LOG(Debug, "() returning " << largest_block);
  return largest_block;
}

std::size_t DynamicPoolMap::getReleasableSize() const noexcept
{
  std::size_t releasable_bytes{0};
  for (auto& rec : m_free_map) {
    const std::size_t bytes{rec.first};
    Pointer ptr;
    bool is_head;
    std::size_t whole_bytes;
    std::tie(ptr, is_head, whole_bytes) = rec.second;
    if (is_head && bytes == whole_bytes)
      releasable_bytes += bytes;
  }

  return releasable_bytes;
}

Platform DynamicPoolMap::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits DynamicPoolMap::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

void DynamicPoolMap::mergeFreeBlocks()
{
  if (m_free_map.size() < 2)
    return;

  using PointerMap = std::map<Pointer, SizeTuple>;

  UMPIRE_LOG(Debug, "() Free blocks before: " << getFreeBlocks());

  // Make a free block map from pointers -> size pairs
  PointerMap free_pointer_map;

  for (auto& rec : m_free_map) {
    const std::size_t bytes{rec.first};
    Pointer ptr;
    bool is_head;
    std::size_t whole_bytes;
    std::tie(ptr, is_head, whole_bytes) = rec.second;
    free_pointer_map.insert(
        std::make_pair(ptr, std::make_tuple(bytes, is_head, whole_bytes)));
  }

  // this map is iterated over from low to high in terms of key = pointer
  // address. Colaesce these...

  auto it = free_pointer_map.begin();
  auto next_it = free_pointer_map.begin();
  ++next_it;
  auto end = free_pointer_map.end();

  while (next_it != end) {
    const unsigned char* this_addr{static_cast<unsigned char*>(it->first)};
    std::size_t this_bytes, this_whole_bytes;
    bool this_is_head;
    std::tie(this_bytes, this_is_head, this_whole_bytes) = it->second;

    const unsigned char* next_addr{static_cast<unsigned char*>(next_it->first)};
    std::size_t next_bytes, next_whole_bytes;
    bool next_is_head;
    std::tie(next_bytes, next_is_head, next_whole_bytes) = next_it->second;

    // Check if we can merge *it and *next_it
    const bool contiguous{this_addr + this_bytes == next_addr};
    if (contiguous && !next_is_head) {
      UMPIRE_ASSERT(this_whole_bytes == next_whole_bytes);
      std::get<0>(it->second) += next_bytes;
      next_it = free_pointer_map.erase(next_it);
    } else {
      ++it;
      ++next_it;
    }
  }

  // Now the external map may have shrunk, so rebuild the original map
  m_free_map.clear();
  for (auto& rec : free_pointer_map) {
    Pointer ptr{rec.first};
    std::size_t bytes, whole_bytes;
    bool is_head;
    std::tie(bytes, is_head, whole_bytes) = rec.second;
    insertFree(ptr, bytes, is_head, whole_bytes);
  }

  UMPIRE_LOG(Debug, "() Free blocks after: " << getFreeBlocks());
}

std::size_t DynamicPoolMap::releaseFreeBlocks()
{
  UMPIRE_LOG(Debug, "()");

  std::size_t released_bytes{0};

  auto it = m_free_map.cbegin();
  auto end = m_free_map.cend();

  while (it != end) {
    const std::size_t bytes{it->first};
    Pointer ptr;
    bool is_head;
    std::size_t whole_bytes;
    std::tie(ptr, is_head, whole_bytes) = it->second;
    if (is_head && bytes == whole_bytes) {
      released_bytes += bytes;
      deallocateBlock(ptr, bytes);
      it = m_free_map.erase(it);
    } else {
      ++it;
    }
  }

#if defined(UMPIRE_ENABLE_BACKTRACE)
  if (released_bytes > 0) {
    umpire::util::backtrace bt;
    umpire::util::backtracer<>::get_backtrace(bt);
    UMPIRE_LOG(
        Info, "actual_size: " << m_actual_bytes
                              << " (prev: " << (m_actual_bytes + released_bytes)
                              << ") " << umpire::util::backtracer<>::print(bt));
  }
#endif

  return released_bytes;
}

void DynamicPoolMap::coalesce()
{
  // Coalesce differs from release in that it puts back a single block of the
  // size it released
  UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \""
                << getName() << "\" }");

  do_coalesce();
}

void DynamicPoolMap::release()
{
  UMPIRE_LOG(Debug, "()");

  // Coalesce first so that we are able to release the most memory possible
  mergeFreeBlocks();

  // Free any blocks with is_head
  releaseFreeBlocks();

  // NOTE This differs from coalesce above in that it does not reallocate a
  // free block to keep actual size the same.
}

void DynamicPoolMap::do_coalesce()
{
  mergeFreeBlocks();
  // Now all possible the free blocks that could be merged have been

  // Only release and create new block if more than one block is present
  if (m_free_map.size() > 1) {
    const std::size_t released_bytes{releaseFreeBlocks()};
    // Deallocated and removed released_bytes from m_free_map

    // If this removed anything from the map, re-allocate a single large chunk
    // and insert to free map
    if (released_bytes > 0) {
      Pointer ptr{allocateBlock(released_bytes)};
      insertFree(ptr, released_bytes, true, released_bytes);
    }
  }
}

DynamicPoolMap::CoalesceHeuristic DynamicPoolMap::percent_releasable(
    int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR("Invalid percentage of "
                 << percentage
                 << ", percentage must be an integer between 0 and 100");
  }

  if (percentage == 0) {
    return [=](const DynamicPoolMap& UMPIRE_UNUSED_ARG(pool)) { return false; };
  } else if (percentage == 100) {
    return [=](const strategy::DynamicPoolMap& pool) {
      return (pool.getActualSize() == pool.getReleasableSize());
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);

    return [=](const strategy::DynamicPoolMap& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold =
          static_cast<std::size_t>(f * pool.getActualSize());
      return (pool.getReleasableSize() >= threshold);
    };
  }
}

} // end of namespace strategy
} // end of namespace umpire
