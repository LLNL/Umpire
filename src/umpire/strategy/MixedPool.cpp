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

#include "umpire/util/Macros.hpp"
#include "umpire/strategy/MixedPool.hpp"

#include <cstdint>
#include <algorithm>

namespace umpire {
namespace strategy {

MixedPool::MixedPool(const std::string& name, int id,
                     Allocator allocator,
                     size_t smallest_fixed_blocksize,
                     size_t largest_fixed_blocksize,
                     size_t max_fixed_pool_size,
                     size_t size_multiplier,
                     size_t dynamic_min_initial_alloc_size,
                     size_t dynamic_min_alloc_size,
                     DynamicPool::Coalesce_Heuristic coalesce_heuristic) noexcept :
  AllocationStrategy(name, id),
  m_map(),
  m_fixed_pool_map(),
  m_fixed_pool(),
  m_dynamic_pool("internal_dynamic_pool", -1, allocator,
                 dynamic_min_initial_alloc_size,
                 dynamic_min_alloc_size,
                 coalesce_heuristic),
  m_allocator(allocator.getAllocationStrategy())
{
  size_t obj_bytes = smallest_fixed_blocksize;
  while (obj_bytes <= largest_fixed_blocksize) {
    size_t obj_per_pool = std::min(64 * sizeof(int) * 8,
                                   static_cast<size_t>(static_cast<float>(obj_bytes) / max_fixed_pool_size));
    if (obj_per_pool > 1) {
      m_fixed_pool.emplace_back("internal_fixed_pool", -1, allocator, obj_bytes, obj_per_pool);
      m_fixed_pool_map.push_back(obj_bytes);
    }
    else {
      break;
    }
    obj_bytes *= size_multiplier;
  }

  if (m_fixed_pool.size() == 0) {
    UMPIRE_LOG(Debug, "Mixed Pool is reverting to a dynamic pool only");
  }
}

void* MixedPool::allocate(size_t bytes)
{
  // Find pool index
  size_t index = 0;
  for (size_t i = 0; i < m_fixed_pool_map.size(); ++i) {
    if (bytes > m_fixed_pool_map[index]) { index++; }
    else { break; }
  }

  void* mem;

  if (index < m_fixed_pool.size()) {
    // allocate in fixed pool
    mem = m_fixed_pool[index].allocate();
    m_map[reinterpret_cast<uintptr_t>(mem)] = index;
  }
  else {
    // allocate in dynamic pool
    mem = m_dynamic_pool.allocate(bytes);
    m_map[reinterpret_cast<uintptr_t>(mem)] = -1;
  }
  return mem;
}

void MixedPool::deallocate(void* ptr)
{
  auto iter = m_map.find(reinterpret_cast<uintptr_t>(ptr));
  if (iter != m_map.end()) {
    const int index = iter->second;
    if (index < 0) {
      m_dynamic_pool.deallocate(ptr);
    }
    else {
      m_fixed_pool[index].deallocate(ptr);
    }
  }
}

void MixedPool::release()
{
  UMPIRE_LOG(Debug, "MixedPool::release(): Not yet implemented");
}

std::size_t MixedPool::getCurrentSize() const noexcept
{
  size_t size = 0;
  for (auto& fp : m_fixed_pool) size += fp.getCurrentSize();
  size += m_dynamic_pool.getCurrentSize();
  return size;
}

std::size_t MixedPool::getActualSize() const noexcept
{
  size_t size = 0;
  for (auto& fp : m_fixed_pool) size += fp.getActualSize();
  size += m_dynamic_pool.getActualSize();
  return size;
}

std::size_t MixedPool::getHighWatermark() const noexcept
{
  size_t size = 0;
  for (auto& fp : m_fixed_pool) size += fp.getHighWatermark();
  size += m_dynamic_pool.getHighWatermark();
  return size;
}

Platform MixedPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

}
}
