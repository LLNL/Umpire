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

static int round_up(std::size_t num, std::size_t factor)
{
  return num + factor - 1 - (num - 1) % factor;
}

namespace umpire {
namespace strategy {

inline static std::size_t alignmentAdjust(const std::size_t size) {
  const std::size_t AlignmentBoundary = 16;
  return std::size_t (size + (AlignmentBoundary-1)) & ~(AlignmentBoundary-1);
}

DynamicPool::DynamicPool(const std::string& name,
                         int id,
                         Allocator allocator,
                         const std::size_t initial_alloc_bytes,
                         const std::size_t min_alloc_bytes,
                         const short align_bytes,
                         Coalesce_Heuristic UMPIRE_UNUSED_ARG(coalesce_heuristic)) noexcept :
  AllocationStrategy(name, id),
  m_allocator{allocator.getAllocationStrategy()},
  m_min_alloc_bytes{min_alloc_bytes},
  m_align_bytes{align_bytes},
  m_used_map{},
  m_free_map{},
  m_curr_bytes{initial_alloc_bytes},
  m_actual_bytes{initial_alloc_bytes},
  m_highwatermark{initial_alloc_bytes}
{
  m_free_map.insert(
    SizeMap::value_type{initial_alloc_bytes, m_allocator->allocate(initial_alloc_bytes)});
}

DynamicPool::~DynamicPool()
{
}

void*
DynamicPool::allocate(std::size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");

  const std::size_t actual_bytes = round_up(bytes, m_align_bytes);
  SizeMap::iterator iter{m_free_map.upper_bound(actual_bytes)};
  Pointer ptr{nullptr};

  if (iter != m_free_map.end()) {
    ptr = iter->second;
    // Add used map
    m_used_map.insert(std::make_pair(ptr, actual_bytes));

    // Remove the entry from the free map
    m_free_map.erase(iter);

    m_curr_bytes += actual_bytes;

    const int64_t left_bytes{static_cast<int64_t>(iter->first - actual_bytes)};
    if (left_bytes > m_align_bytes) {
      // Add to free map
      m_free_map.insert(SizeMap::value_type{left_bytes, static_cast<unsigned char*>(ptr) + actual_bytes});
    }
  } else {
    // Allocate new chunk -- note that this does not check whether alignment is met
    if (actual_bytes > m_min_alloc_bytes) {
      ptr = m_allocator->allocate(actual_bytes);

      // Add used
      m_used_map.insert(std::make_pair(ptr, actual_bytes));

      m_actual_bytes += actual_bytes;
      m_curr_bytes += actual_bytes;
    } else {
      ptr = m_allocator->allocate(m_min_alloc_bytes);

      m_actual_bytes += m_min_alloc_bytes;

      // Add used
      m_used_map.insert(std::make_pair(ptr, actual_bytes));

      m_curr_bytes += actual_bytes;

      // Add free
      const int64_t left_bytes{static_cast<int64_t>(m_min_alloc_bytes - actual_bytes)};
      if (left_bytes > m_align_bytes)
        m_free_map.insert(std::make_pair(left_bytes, static_cast<unsigned char*>(ptr) + actual_bytes));
    }
  }

  return ptr;
}

void
DynamicPool::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  UMPIRE_ASSERT(ptr);

  auto iter = m_used_map.find(ptr);

  if (iter->second) {
    // Fast way to check if key was found

    // Insert in free map
    const std::size_t bytes{*iter->second};
    m_free_map.insert(SizeMap::value_type{bytes, iter->first});

    // remove from used map
    m_used_map.erase(iter);

    m_curr_bytes -= bytes;
  } else {
    UMPIRE_ERROR("Cound not found ptr = " << ptr);
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

  auto iter = m_free_map.begin();
  auto end = m_free_map.end();

  while (iter != end) {
    releasable_bytes += iter->first;
    ++iter;
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

Platform
DynamicPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void DynamicPool::coalesce() noexcept
{
  UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \"" << getName() << "\" }");

  using PointerMap = std::map<Pointer, std::size_t>;

  // Reverse the free chunk map
  PointerMap free_pointer_map;

  for (auto& rec : m_free_map) {
    free_pointer_map.insert(std::make_pair(rec.second, rec.first));
  }

  // this map is iterated over from low to high in terms of key = pointer address.
  // Colaesce these...

  for (auto it = free_pointer_map.rbegin(), next_it = it; it != free_pointer_map.rend(); it = next_it) {
    --next_it;
    if ((next_it != free_pointer_map.rend()) &&
        (static_cast<unsigned char*>(next_it->first) + next_it->second == it->first)) {
      free_pointer_map.erase(std::next(it).base());
      next_it->second += it->second;
    }
  }

  // Now the external map may have shrunk, so rebuild the original map
  m_free_map.clear();
  for (auto& rec : free_pointer_map) {
    m_free_map.insert(std::make_pair(rec.second, rec.first));
  }
}

void
DynamicPool::release()
{
  for (auto& rec : m_free_map) {
    m_curr_bytes -= rec.first;
    m_actual_bytes -= rec.first;
    m_allocator->deallocate(rec.second);
  }

  m_free_map.clear();
}

} // end of namespace strategy
} // end of namespace umpire
