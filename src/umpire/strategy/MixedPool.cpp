//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/Macros.hpp"
#include "umpire/strategy/MixedPool.hpp"

#include <cstdint>
#include <algorithm>

namespace umpire {
namespace strategy {

MixedPool::MixedPool(const std::string& name, int id,
                     Allocator allocator,
                     std::size_t smallest_fixed_blocksize,
                     std::size_t largest_fixed_blocksize,
                     std::size_t max_fixed_pool_size,
                     std::size_t size_multiplier,
                     const std::size_t dynamic_initial_alloc_size,
                     const std::size_t dynamic_min_alloc_size,
                     DynamicPool::CoalesceHeuristic coalesce_heuristic,
                     const int dynamic_align_bytes) noexcept :
  AllocationStrategy(name, id),
  m_map(),
  m_fixed_pool_map(),
  m_fixed_pool(),
  m_dynamic_pool("internal_dynamic_pool", -1, allocator,
                 dynamic_initial_alloc_size,
                 dynamic_min_alloc_size,
                 coalesce_heuristic,
                 dynamic_align_bytes),
  m_allocator(allocator.getAllocationStrategy())
{
  std::size_t obj_bytes = smallest_fixed_blocksize;
  while (obj_bytes <= largest_fixed_blocksize) {
    std::size_t obj_per_pool = std::min(64 * sizeof(int) * 8,
                                   static_cast<std::size_t>(static_cast<float>(obj_bytes) / max_fixed_pool_size));
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

void* MixedPool::allocate(std::size_t bytes)
{
  // Find pool index
  int index = 0;
  for (std::size_t i = 0; i < m_fixed_pool_map.size(); ++i) {
    if (bytes > m_fixed_pool_map[index]) { index++; }
    else { break; }
  }

  void* mem;

  if (static_cast<std::size_t>(index) < m_fixed_pool.size()) {
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
  std::size_t size = 0;
  for (auto& fp : m_fixed_pool) size += fp.getCurrentSize();
  size += m_dynamic_pool.getCurrentSize();
  return size;
}

std::size_t MixedPool::getActualSize() const noexcept
{
  std::size_t size = 0;
  for (auto& fp : m_fixed_pool) size += fp.getActualSize();
  size += m_dynamic_pool.getActualSize();
  return size;
}

std::size_t MixedPool::getHighWatermark() const noexcept
{
  std::size_t size = 0;
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
