//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/MixedPool.hpp"

#include <algorithm>
#include <cstdint>

#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace strategy {

MixedPool::MixedPool(const std::string& name, int id, Allocator allocator,
                     std::size_t smallest_fixed_obj_size,
                     std::size_t largest_fixed_obj_size,
                     std::size_t max_initial_fixed_pool_size,
                     std::size_t fixed_size_multiplier,
                     const std::size_t quick_pool_initial_alloc_size,
                     const std::size_t quick_pool_min_alloc_size,
                     const std::size_t quick_pool_align_bytes,
                     QuickPool::CoalesceHeuristic should_coalesce) noexcept
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "MixedPool"},
      m_map{},
      m_fixed_pool_map{},
      m_fixed_pool{},
      m_quick_pool{"internal_quick_pool",
                     -1,
                     allocator,
                     quick_pool_initial_alloc_size,
                     quick_pool_min_alloc_size,
                     quick_pool_align_bytes,
                     should_coalesce},
      m_allocator{allocator.getAllocationStrategy()}
{
  std::size_t obj_size{smallest_fixed_obj_size};
  while (obj_size <= largest_fixed_obj_size) {
    const std::size_t obj_per_pool{
        std::min(64 * sizeof(int) * 8, max_initial_fixed_pool_size / obj_size)};
    if (obj_per_pool > 1) {
      m_fixed_pool.emplace_back(util::make_unique<FixedPool>(
          "internal_fixed_pool", -1, allocator, obj_size, obj_per_pool));
      m_fixed_pool_map.emplace_back(obj_size);
    } else {
      break;
    }
    obj_size *= fixed_size_multiplier;
  }

  if (m_fixed_pool.size() == 0) {
    UMPIRE_LOG(Debug, "Mixed Pool is reverting to a quick pool");
  }
}

void* MixedPool::allocate(std::size_t bytes)
{
  // Find pool index
  int index = 0;
  for (std::size_t i = 0; i < m_fixed_pool_map.size(); ++i) {
    if (bytes > m_fixed_pool_map[index]) {
      index++;
    } else {
      break;
    }
  }

  void* mem;

  if (static_cast<std::size_t>(index) < m_fixed_pool.size()) {
    // allocate in fixed pool
    mem = m_fixed_pool[index]->allocate_internal(m_fixed_pool_map[index]);
    m_map[reinterpret_cast<uintptr_t>(mem)] = index;
  } else {
    // allocate in quick pool
    mem = m_quick_pool.allocate_internal(bytes);
    m_map[reinterpret_cast<uintptr_t>(mem)] = -1;
  }
  return mem;
}

void MixedPool::deallocate(void* ptr, std::size_t size)
{
  auto iter = m_map.find(reinterpret_cast<uintptr_t>(ptr));
  if (iter != m_map.end()) {
    const int index = iter->second;
    if (index < 0) {
      m_quick_pool.deallocate_internal(ptr, size);
    } else {
      m_fixed_pool[index]->deallocate_internal(ptr, size);
    }
  }
}

void MixedPool::release()
{
  UMPIRE_LOG(Debug, "MixedPool::release(): Not yet implemented");
}

std::size_t MixedPool::getActualSize() const noexcept
{
  std::size_t size = 0;
  for (auto& fp : m_fixed_pool)
    size += fp->getActualSize();
  size += m_quick_pool.getActualSize();
  return size;
}

Platform MixedPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits MixedPool::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

} // namespace strategy
} // namespace umpire
