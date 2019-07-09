//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/DynamicPool.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"

#include <cstdlib>
#include <algorithm>
#include <sstream>

inline static std::size_t round_up(std::size_t num, std::size_t factor)
{
  return num + factor - 1 - (num - 1) % factor;
}

namespace umpire {
namespace strategy {

DynamicPool::DynamicPool(const std::string& name,
                         int id,
                         Allocator allocator,
                         const std::size_t initial_alloc_bytes,
                         const std::size_t min_alloc_bytes,
                         CoalesceHeuristic coalesce_heuristic,
                         const int align_bytes) noexcept :
  AllocationStrategy(name, id),
  m_allocator{allocator.getAllocationStrategy()},
  m_min_alloc_bytes{min_alloc_bytes},
  m_align_bytes{align_bytes},
  m_coalesce_heuristic{coalesce_heuristic},
  m_used_map{},
  m_free_map{},
  m_curr_bytes{0},
  m_actual_bytes{initial_alloc_bytes},
  m_highwatermark{initial_alloc_bytes}
{
  insertFree(m_allocator->allocate(initial_alloc_bytes),
             initial_alloc_bytes, true, initial_alloc_bytes);
}

DynamicPool::~DynamicPool()
{
  // Coalesce to get whole blocks if possible
  doCoalesce();

  // Error if blocks are still in use
  std::size_t num_heads = 0;
  for (auto& rec : m_used_map) {
    std::size_t bytes, whole_bytes;
    bool is_head;
    if (rec.second) {
      std::tie(bytes, is_head, whole_bytes) = *rec.second;
      if (is_head) ++num_heads;
    }
  }
  if (m_used_map.size() > 0) {
    std::stringstream ss;
    ss << "Pointers still in use at destruction time.";
    if (num_heads > 0) {
      ss << " This will cause " << num_heads << " leak(s).";
    }
    UMPIRE_LOG(Error, ss.str());
  }

  // Free any unused blocks
  for (auto& rec : m_free_map) {
    const std::size_t bytes{rec.first};
    void* addr;
    bool is_head;
    std::size_t whole_bytes;
    std::tie(addr, is_head, whole_bytes) = rec.second;
    // Deallocate if this is a whole block
    if (is_head && bytes == whole_bytes) m_allocator->deallocate(addr);
  }
}

void DynamicPool::insertUsed(Pointer addr, std::size_t bytes, bool is_head, std::size_t whole_bytes)
{
  m_used_map.insert(std::make_pair(addr, std::make_tuple(bytes, is_head, whole_bytes)));
}

void DynamicPool::insertFree(Pointer addr, std::size_t bytes, bool is_head, std::size_t whole_bytes)
{
  m_free_map.insert(std::make_pair(bytes, std::make_tuple(addr, is_head, whole_bytes)));
}

DynamicPool::SizeMap::const_iterator DynamicPool::findFreeBlock(std::size_t bytes) const
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

void* DynamicPool::allocate(std::size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");

  const std::size_t actual_bytes = round_up(bytes, m_align_bytes);
  Pointer ptr{nullptr};

  // Check if the previous block is a match
  SizeMap::const_iterator iter{findFreeBlock(actual_bytes)};

  // This is optional, but it might help the growth of the pool...
  if (iter == m_free_map.end()) {
    doCoalesce();
    iter = findFreeBlock(actual_bytes);
  }

  if (iter != m_free_map.end()) {
    // Found this acceptable address pair
    bool is_head;
    std::size_t whole_bytes;
    std::tie(ptr, is_head, whole_bytes) = iter->second;

    // Add used map
    insertUsed(ptr, actual_bytes, is_head, whole_bytes);

    // Remove the entry from the free map
    const std::size_t free_size = iter->first;
    m_free_map.erase(iter);

    m_curr_bytes += actual_bytes;

    const int64_t left_bytes{static_cast<int64_t>(free_size - actual_bytes)};
    if (left_bytes > m_align_bytes) {
      insertFree(static_cast<unsigned char*>(ptr) + actual_bytes, left_bytes, false, whole_bytes);
    }
  } else {
    // Allocate new block -- note that this does not check whether alignment is met
    if (actual_bytes > m_min_alloc_bytes) {
      ptr = m_allocator->allocate(actual_bytes);

      // Add used
      insertUsed(ptr, actual_bytes, true, actual_bytes);

      m_actual_bytes += actual_bytes;
      m_curr_bytes += actual_bytes;
    } else {
      ptr = m_allocator->allocate(m_min_alloc_bytes);
      m_actual_bytes += m_min_alloc_bytes;

      // Add used
      insertUsed(ptr, actual_bytes, true, m_min_alloc_bytes);
      m_curr_bytes += actual_bytes;

      // Add free
      const int64_t left_bytes{static_cast<int64_t>(m_min_alloc_bytes - actual_bytes)};
      if (left_bytes > m_align_bytes)
        insertFree(static_cast<unsigned char*>(ptr) + actual_bytes, left_bytes, false, m_min_alloc_bytes);
    }
  }

  if (m_curr_bytes > m_highwatermark) m_highwatermark = m_curr_bytes;

  return ptr;
}

void DynamicPool::deallocate(void* ptr)
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

    // remove from used map
    m_used_map.erase(iter);

    m_curr_bytes -= bytes;
  } else {
    UMPIRE_ERROR("Cound not found ptr = " << ptr);
  }

  if (m_coalesce_heuristic(*this)) {
    UMPIRE_LOG(Debug, this << " heuristic function returned true, calling coalesce()");
    coalesce();
  }
}

std::size_t DynamicPool::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_curr_bytes);
  return m_curr_bytes;
}

std::size_t DynamicPool::getActualSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_actual_bytes);
  return m_actual_bytes;
}

std::size_t DynamicPool::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << m_highwatermark);
  return m_highwatermark;
}

std::size_t DynamicPool::getFreeBlocks() const noexcept
{
  return m_free_map.size();
}

std::size_t DynamicPool::getInUseBlocks() const noexcept
{
  return m_used_map.size();
}

std::size_t DynamicPool::getBlocksInPool() const noexcept
{
  const std::size_t total_blocks{getFreeBlocks() + getInUseBlocks()};
  UMPIRE_LOG(Debug, "() returning " << total_blocks);
  return total_blocks;
}

std::size_t DynamicPool::getReleasableSize() const noexcept
{
  std::size_t releasable_bytes{0};
  for (auto& rec : m_free_map) {
    const std::size_t bytes{rec.first};
    Pointer ptr;
    bool is_head;
    std::size_t whole_bytes;
    std::tie(ptr, is_head, whole_bytes) = rec.second;
    if (is_head && bytes == whole_bytes) releasable_bytes += bytes;
  }

  return releasable_bytes;
}

Platform DynamicPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void DynamicPool::doCoalesce()
{
  using PointerMap = std::map<Pointer, SizeTuple>;

  // Make a free block map from pointers -> size pairs
  PointerMap free_pointer_map;

  for (auto& rec : m_free_map) {
    const std::size_t bytes{rec.first};
    Pointer ptr;
    bool is_head;
    std::size_t whole_bytes;
    std::tie(ptr, is_head, whole_bytes) = rec.second;
    free_pointer_map.insert(std::make_pair(ptr, std::make_tuple(bytes, is_head, whole_bytes)));
  }

  if (free_pointer_map.size() < 2) return;

  // this map is iterated over from low to high in terms of key = pointer address.
  // Colaesce these...

  auto it = free_pointer_map.begin();
  auto next_it = free_pointer_map.begin(); ++next_it;
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
}

std::size_t DynamicPool::doRelease()
{
  UMPIRE_LOG(Debug, "()");

  // Coalesce first so that we are able to release the most memory possible
  doCoalesce();

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
      m_actual_bytes -= bytes;
      m_allocator->deallocate(ptr);
      it = m_free_map.erase(it);
    } else {
      ++it;
    }
  }

  return released_bytes;
}

void DynamicPool::coalesce()
{
  // Coalesce differs from release in that it puts back a single block of the size it released
  UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \"" << getName() << "\" }");

  doCoalesce();
  // Now all possible the free blocks that could be merged have been

  // Only release and create new block if more than one block is present
  if (m_free_map.size() > 1) {
    const std::size_t released_bytes{doRelease()};
    // Deallocated and removed released_bytes from m_free_map

    // If this removed anything from the map, re-allocate a single large chunk
    if (released_bytes > 0) {
      const std::size_t actual_bytes{round_up(released_bytes, m_align_bytes)};

      const Pointer ptr{m_allocator->allocate(actual_bytes)};
      m_actual_bytes += actual_bytes;

      // Add used
      insertFree(ptr, actual_bytes, true, actual_bytes);
    }
  }
}

void DynamicPool::release()
{
  UMPIRE_LOG(Debug, "()");

  // Coalesce first so that we are able to release the most memory possible
  doCoalesce();
  doRelease();

  // This differs from coalesce above in that it does not reallocate a
  // free block to keep actual size the same.
}

} // end of namespace strategy
} // end of namespace umpire
