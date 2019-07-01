//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MixedPool_HPP
#define UMPIRE_MixedPool_HPP

#include <memory>
#include <vector>
#include <map>

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/FixedPool.hpp"

#include "umpire/strategy/DynamicPoolHeuristic.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {
namespace strategy {

/**
 * \brief A faster pool that pulls from a series of pools
 *
 * Pool implementation using a series of FixedPools for small sizes,
 * and a DynamicPool for sizes larger than (1 << LastFixed) bytes.
 */
class MixedPool :
  public AllocationStrategy
{
  public:
    MixedPool(
      const std::string& name, int id,
      Allocator allocator,
      std::size_t smallest_fixed_blocksize = (1 << 8), // 256B
      std::size_t largest_fixed_blocksize = (1 << 17), // 1024K
      std::size_t max_fixed_pool_size = 1024*1024 * 2, // 2MB
      std::size_t size_multiplier = 10,                // 10x over previous size
      std::size_t dynamic_min_initial_alloc_size = (512 * 1024 * 1024),
      std::size_t dynamic_min_alloc_size = (1 * 1024 *1024),
      DynamicPool::Coalesce_Heuristic coalesce_heuristic = heuristic_percent_releasable(100)
      ) noexcept;

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr) override;

    void release() override;

    std::size_t getCurrentSize() const noexcept override;
    std::size_t getActualSize() const noexcept override;
    std::size_t getHighWatermark() const noexcept override;

    Platform getPlatform() noexcept override;

  private:
    using IntMap = std::map<uintptr_t, int>;
    IntMap m_map;
    std::vector<std::size_t> m_fixed_pool_map;
    std::vector<FixedPool> m_fixed_pool;
    DynamicPool m_dynamic_pool;
    AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_MixedPool_HPP
