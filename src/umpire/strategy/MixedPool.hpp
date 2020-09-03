//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MixedPool_HPP
#define UMPIRE_MixedPool_HPP

#include <map>
#include <memory>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"
#include "umpire/strategy/FixedPool.hpp"

namespace umpire {
namespace strategy {

/**
 * \brief A faster pool that pulls from a series of pools
 *
 * Pool implementation using a series of FixedPools for small sizes,
 * and a DynamicPool for sizes larger than (1 << LastFixed) bytes.
 */
class MixedPool : public AllocationStrategy {
 public:
  /**
   * \brief Creates a MixedPool of one or more fixed pools and a dynamic pool
   * for large allocations.
   *
   * \param name Name of the pool
   * \param id Unique identifier for lookup later in ResourceManager
   * \param allocator Underlying allocator
   * \param smallest_fixed_obj_size Smallest fixed pool object size in bytes
   * \param largest_fixed_obj_size Largest fixed pool object size in bytes
   * \param max_initial_fixed_pool_size Largest initial size of any fixed pool
   * \param fixed_size_multiplier Fixed pool object size increase factor
   * \param dynamic_initial_alloc_size Size the dynamic pool initially allocates
   * \param dynamic_min_alloc_bytes Minimum size of all future allocations in
   * the dynamic pool \param dynamic_align_bytes Size with which to align
   * allocations (for the dynamic pool) \param should_coalesce Heuristic
   * callback function (for the dynamic pool)
   */
  MixedPool(const std::string& name, int id, Allocator allocator,
            std::size_t smallest_fixed_obj_size = (1 << 8),            // 256B
            std::size_t largest_fixed_obj_size = (1 << 17),            // 1024K
            std::size_t max_initial_fixed_pool_size = 1024 * 1024 * 2, // 2MB
            std::size_t fixed_size_multiplier = 16, // 16x over previous size
            const std::size_t dynamic_initial_alloc_size = (512 * 1024 * 1024),
            const std::size_t dynamic_min_alloc_size = (1 * 1024 * 1024),
            const std::size_t dynamic_align_bytes = 16,
            DynamicPoolMap::CoalesceHeuristic should_coalesce =
                DynamicPoolMap::percent_releasable(100)) noexcept;

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr) override;

  void release() override;

  std::size_t getCurrentSize() const noexcept override;
  std::size_t getActualSize() const noexcept override;
  std::size_t getHighWatermark() const noexcept override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  using IntMap = std::map<uintptr_t, int>;
  IntMap m_map;
  std::vector<std::size_t> m_fixed_pool_map;
  std::vector<std::unique_ptr<FixedPool>> m_fixed_pool;
  DynamicPoolMap m_dynamic_pool;
  AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_MixedPool_HPP
