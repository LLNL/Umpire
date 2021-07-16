//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/strategy/dynamic_pool_map.hpp"
#include "umpire/strategy/fixed_pool.hpp"

#include "camp/make_unique.hpp"

#include <memory>
#include <vector>
#include <map>


namespace umpire {
namespace strategy {

/**
 * \brief A faster pool that pulls from a series of pools
 *
 * Pool implementation using a series of FixedPools for small sizes,
 * and a DynamicPool for sizes larger than (1 << LastFixed) bytes.
 */
template <typename Memory=memory, bool Tracking=true, typename FixedPool=fixed_pool<Memory>, typename DynamicPool=dynamic_pool_map<Memory>>
class mixed_pool :
  public allocation_strategy
{
  public:
  /**
   * \brief Creates a MixedPool of one or more fixed pools and a dynamic pool
   * for large allocations.
   *
   * \param name Name of the pool
   * \param allocator Underlying allocator
   * \param smallest_fixed_obj_size Smallest fixed pool object size in bytes
   * \param largest_fixed_obj_size Largest fixed pool object size in bytes
   * \param max_initial_fixed_pool_size Largest initial size of any fixed pool
   * \param fixed_size_multiplier Fixed pool object size increase factor
   * \param dynamic_initial_alloc_size Size the dynamic pool initially allocates
   * \param dynamic_min_alloc_bytes Minimum size of all future allocations in the dynamic pool
   * \param dynamic_align_bytes Size with which to align allocations (for the dynamic pool)
   * \param coalesce_heuristic Heuristic callback function (for the dynamic pool)
   */
    mixed_pool(
      const std::string& name,
      Memory* allocator,
      std::size_t smallest_fixed_obj_size = (1 << 8),          // 256B
      std::size_t largest_fixed_obj_size = (1 << 17),          // 1024K
      std::size_t max_initial_fixed_pool_size = 1024*1024 * 2, // 2MB
      std::size_t fixed_size_multiplier = 16,                  // 16x over previous size
      const std::size_t dynamic_initial_alloc_size = (512 * 1024 * 1024),
      const std::size_t dynamic_min_alloc_size = (1 * 1024 *1024),
      const std::size_t dynamic_align_bytes = 16,
      typename DynamicPool::CoalesceHeuristic dynamic_coalesce_heuristic = DynamicPool::percent_releasable(100)) noexcept :
    allocation_strategy{name},
    m_map{},
    m_fixed_pool_map{},
    m_fixed_pool{},
    m_dynamic_pool{"internal_dynamic_pool",
                 allocator,
                 dynamic_initial_alloc_size,
                 dynamic_min_alloc_size,
                 dynamic_align_bytes,
                 dynamic_coalesce_heuristic},
    m_allocator{allocator}
    {
      std::size_t obj_size{smallest_fixed_obj_size};
      while (obj_size <= largest_fixed_obj_size) {
        const std::size_t obj_per_pool{
          std::min(64 * sizeof(int) * 8, max_initial_fixed_pool_size / obj_size)};
        if (obj_per_pool > 1) {
          m_fixed_pool.emplace_back(camp::make_unique<FixedPool>("internal_fixed_pool", allocator, obj_size, obj_per_pool));
          m_fixed_pool_map.emplace_back(obj_size);
        } else {
          break;
        }
        obj_size *= fixed_size_multiplier;
      }

      if (m_fixed_pool.size() == 0) {
        UMPIRE_LOG(Debug, "Mixed Pool is reverting to a dynamic pool");
      }
    }

    void* allocate(std::size_t bytes) override
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
        mem = m_fixed_pool[index]->allocate();
        m_map[reinterpret_cast<uintptr_t>(mem)] = index;
      }
      else {
        // allocate in dynamic pool
        mem = m_dynamic_pool.allocate(bytes);
        m_map[reinterpret_cast<uintptr_t>(mem)] = -1;
      }
      return mem;
    }

    void deallocate(void* ptr) override
    {
      auto iter = m_map.find(reinterpret_cast<uintptr_t>(ptr));
      if (iter != m_map.end()) {
        const int index = iter->second;
        if (index < 0) {
          m_dynamic_pool.deallocate(ptr);
        }
        else {
          m_fixed_pool[index]->deallocate(ptr);
        }
      }
    }

    std::size_t get_actual_size() const noexcept override
    {
      std::size_t size = 0;
      for (auto& fp : m_fixed_pool) size += fp->get_actual_size();
      size += m_dynamic_pool.get_actual_size();
      return size;
    }

    camp::resources::Platform get_platform() noexcept override
    {
      return m_allocator->get_platform();
    }

  private:
    using IntMap = std::map<uintptr_t, int>;
    IntMap m_map;
    std::vector<std::size_t> m_fixed_pool_map;
    std::vector<std::unique_ptr<FixedPool>> m_fixed_pool;
    DynamicPool m_dynamic_pool;
    Memory* m_allocator;
};

} // end of namespace strategy
} // end namespace umpire
