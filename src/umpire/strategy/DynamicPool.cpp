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

#include <algorithm>
#include <tuple>

static std::size_t round_up(std::size_t num, std::size_t factor)
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
                         const int align_bytes,
                         Coalesce_Heuristic coalesce_heuristic) noexcept :
  AllocationStrategy(name, id),
  m_allocator{allocator.getAllocationStrategy()},
  m_min_alloc_bytes{min_alloc_bytes},
  m_align_bytes{align_bytes},
  m_do_coalesce{coalesce_heuristic},
  m_used_map{},
  m_free_map{},
  m_curr_bytes{0},
  m_actual_bytes{initial_alloc_bytes},
  m_highwatermark{initial_alloc_bytes}
{
  insertFree(m_allocator->allocate(initial_alloc_bytes), initial_alloc_bytes, true);
}

DynamicPool::~DynamicPool()
{
  // Free any unused chunks
  for (auto& rec : m_free_map) {
    void* addr;
    bool is_head;
    std::tie(addr, is_head) = rec.second;
    // Deallocate if this is a head
    if (is_head) m_allocator->deallocate(addr);
  }
}

void DynamicPool::insertUsed(Pointer addr, std::size_t bytes, bool is_head)
{
  m_used_map.insert(std::make_pair(addr, std::make_pair(bytes, is_head)));
}

void DynamicPool::insertFree(Pointer addr, std::size_t bytes, bool is_head)
{
  m_free_map.insert(std::make_pair(bytes, std::make_pair(addr, is_head)));
}

void* DynamicPool::allocate(std::size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");

  const std::size_t actual_bytes = round_up(bytes, m_align_bytes);
  SizeMap::iterator iter{m_free_map.upper_bound(actual_bytes)};
  Pointer ptr{nullptr};

  // Check if the previous element is a match
  if (iter != m_free_map.begin()) {
    // Back up iterator
    --iter;
    const std::size_t test_bytes{iter->first};
    if (test_bytes < actual_bytes) {
      // Too small, reset iterator to original upper_bound value
      ++iter;
    }
  }

  if (iter != m_free_map.end()) {
    // Found this acceptable address pair
    bool is_head;
    std::tie(ptr, is_head) = iter->second;

    // Add used map
    insertUsed(ptr, actual_bytes, is_head);

    // Remove the entry from the free map
    m_free_map.erase(iter);

    m_curr_bytes += actual_bytes;

    const int64_t left_bytes{static_cast<int64_t>(iter->first - actual_bytes)};
    if (left_bytes > m_align_bytes) {
      insertFree(static_cast<unsigned char*>(ptr) + actual_bytes, left_bytes, false);
    }
  } else {
    // Allocate new chunk -- note that this does not check whether alignment is met
    if (actual_bytes > m_min_alloc_bytes) {
      ptr = m_allocator->allocate(actual_bytes);

      // Add used
      insertUsed(ptr, actual_bytes, true);

      m_actual_bytes += actual_bytes;
      m_curr_bytes += actual_bytes;
    } else {
      ptr = m_allocator->allocate(m_min_alloc_bytes);
      m_actual_bytes += m_min_alloc_bytes;

      // Add used
      insertUsed(ptr, actual_bytes, true);
      m_curr_bytes += actual_bytes;

      // Add free
      const int64_t left_bytes{static_cast<int64_t>(m_min_alloc_bytes - actual_bytes)};
      if (left_bytes > m_align_bytes)
        insertFree(static_cast<unsigned char*>(ptr) + actual_bytes, left_bytes, false);
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
    std::tie(bytes, is_head) = *iter->second;

    // Insert in free map
    insertFree(ptr, bytes, is_head);

    // remove from used map
    m_used_map.erase(iter);

    m_curr_bytes -= bytes;
  } else {
    UMPIRE_ERROR("Cound not found ptr = " << ptr);
  }

  if ( m_do_coalesce(*this) ) {
    UMPIRE_LOG(Debug, this << " Heuristic returned true, calling coalesce()");
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

std::size_t DynamicPool::getBlocksInPool() const noexcept
{
  const std::size_t total_blocks{m_used_map.size() + m_free_map.size()};
  UMPIRE_LOG(Debug, "() returning " << total_blocks);
  return total_blocks;
}

std::size_t DynamicPool::getReleasableSize() const noexcept
{
  std::size_t releasable_bytes{0};

  for (auto& rec : m_free_map) {
    const std::size_t bytes{rec.first};
    const bool is_head{rec.second.second};
    if (is_head) releasable_bytes += bytes;
  }

  return releasable_bytes;
}

std::size_t DynamicPool::getFreeBlocks() const
{
  return m_free_map.size();
}

std::size_t DynamicPool::getInUseBlocks() const
{
  return m_used_map.size();
}

Platform DynamicPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void DynamicPool::doCoalesce()
{
  UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \"" << getName() << "\" }");

  using PointerMap = std::map<Pointer, SizePair>;

  // Reverse the free chunk map
  PointerMap free_pointer_map;

  for (auto& rec : m_free_map) {
    const std::size_t bytes{rec.first};
    Pointer ptr;
    bool is_head;
    std::tie(ptr, is_head) = rec.second;
    free_pointer_map.insert(std::make_pair(ptr, std::make_pair(bytes, is_head)));
  }

  if (free_pointer_map.size() < 2) return;

  // this map is iterated over from low to high in terms of key = pointer address.
  // Colaesce these...

  auto it = free_pointer_map.rbegin();
  auto next_it = it; next_it++;
  auto end = free_pointer_map.rend();

  while (next_it != end) {
    const bool contiguous{static_cast<unsigned char*>(next_it->first) + next_it->second.first == it->first};
    const bool was_head{it->second.second};
    if (contiguous && !was_head) {
      next_it->second.first += it->second.first;
      // The std::next(it).base() is needed because it is a reverse iterator
      free_pointer_map.erase(std::next(it).base());
    }
    it = next_it;
    ++next_it;
  }

  // Now the external map may have shrunk, so rebuild the original map
  m_free_map.clear();
  for (auto& rec : free_pointer_map) {
    Pointer ptr{rec.first};
    std::size_t bytes;
    bool is_head;
    std::tie(bytes, is_head) = rec.second;
    insertFree(ptr, bytes, is_head);
  }
}

void DynamicPool::coalesce()
{
  doCoalesce();

  // Coalesce should add back a single chunk to keep m_actual_bytes the same as before
  const std::size_t released_bytes{doRelease()};

  if (released_bytes > 0) {
    const std::size_t actual_bytes{round_up(released_bytes, m_align_bytes)};

    Pointer ptr{m_allocator->allocate(actual_bytes)};
    m_actual_bytes += actual_bytes;

    // Add used
    insertFree(ptr, actual_bytes, true);
  }
}

std::size_t DynamicPool::doRelease()
{
  UMPIRE_LOG(Debug, "()");

  // Coalesce first so that we are able to release the most memory possible
  doCoalesce();

  std::size_t released_bytes{0};

  for (auto it = m_free_map.cbegin(), next_it = it; it != m_free_map.cend(); it = next_it) {
    ++next_it;

    std::size_t bytes{it->first};
    Pointer ptr;
    bool is_head;
    std::tie(ptr, is_head) = it->second;
    if (is_head) {
      released_bytes += bytes;
      m_actual_bytes -= bytes;
      m_allocator->deallocate(ptr);
      m_free_map.erase(it);
    }
  }

  return released_bytes;
}

void DynamicPool::release()
{
  UMPIRE_LOG(Debug, "()");

  // Coalesce first so that we are able to release the most memory possible
  doCoalesce();

  doRelease();
}

} // end of namespace strategy
} // end of namespace umpire
