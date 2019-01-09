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
#ifndef UMPIRE_DynamicPool_HPP
#define UMPIRE_DynamicPool_HPP

#include <memory>
#include <vector>
#include <functional>

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/Allocator.hpp"

#include "umpire/tpl/simpool/DynamicSizePool.hpp"

namespace umpire {
namespace strategy {

class DynamicPool;
static inline bool default_heuristic( const strategy::DynamicPool& /* strat */ ) { return false; }

/*!
 * \brief Simple dynamic pool for allocations
 *
 * This AllocationStrategy uses Simpool to provide pooling for allocations of
 * any size. The behavior of the pool can be controlled by two parameters: the
 * initial allocation size, and the minimum allocation size.
 *
 * The initial size controls how large the first piece of memory allocated is,
 * and the minimum size controls the lower bound on all future chunk
 * allocations.
 */
class DynamicPool :
  public AllocationStrategy
{
  /*!
   * \brief Callback Heuristic to trigger coalesce of free blocks in pool.
   *
   * The registered heuristic callback function will be called in the following
   * two places:
   *
   * 1. Immediately after a deallocation() has completed.
   * 2. Immediately before an allocation() occurrs.
   */
  using Coalesce_Heuristic = std::function<bool( const strategy::DynamicPool& )>;

  public:
    /*!
     * \brief Construct a new DynamicPool.
     *
     * \param name Name of this instance of the DynamicPool.
     * \param id Id of this instance of the DynamicPool.
     * \param min_initial_alloc_size The minimum size of the first allocation
     *                               the pool will make.
     * \param min_alloc_size The minimum size of all future allocations.
     * \param h_fun Heuristic callback function.
     */
    DynamicPool(
        const std::string& name,
        int _id,
        Allocator allocator,
        const std::size_t min_initial_alloc_size = (512 * 1024 * 1024),
        const std::size_t min_alloc_size = (1 * 1024 *1024),
        Coalesce_Heuristic h_fun = default_heuristic) noexcept;

    void* allocate(size_t bytes) override;

    void deallocate(void* ptr) override;

    void release() override;

    long getCurrentSize() noexcept override;
    long getHighWatermark() noexcept override;
    long getActualSize() noexcept override;

    Platform getPlatform() noexcept override;

    void coalesce() noexcept;

  private:
    DynamicSizePool<>* dpa;

    std::shared_ptr<umpire::strategy::AllocationStrategy> m_allocator;
    Coalesce_Heuristic do_coalesce;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPool_HPP
