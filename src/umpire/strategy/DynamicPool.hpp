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
#ifndef UMPIRE_DynamicPool_HPP
#define UMPIRE_DynamicPool_HPP

#include <memory>
#include <vector>
#include <functional>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"

#include "umpire/Allocator.hpp"

#include "umpire/tpl/simpool/DynamicSizePool.hpp"

namespace umpire {
namespace strategy {

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
  public:
    /*!
     * \brief Callback Heuristic to trigger coalesce of free blocks in pool.
     *
     * The registered heuristic callback function will be called immediately
     * after a deallocation() has completed from the pool.
     */
    using Coalesce_Heuristic = std::function<bool( const strategy::DynamicPool& )>;

    /*!
     * \brief Construct a new DynamicPool.
     *
     * \param name Name of this instance of the DynamicPool.
     * \param id Id of this instance of the DynamicPool.
     * \param min_initial_alloc_size The minimum size of the first allocation
     *                               the pool will make.
     * \param min_alloc_size The minimum size of all future allocations.
     * \param coalesce_heuristic Heuristic callback function.
     */
    DynamicPool(
        const std::string& name,
        int id,
        Allocator allocator,
        const std::size_t min_initial_alloc_size = (512 * 1024 * 1024),
        const std::size_t min_alloc_size = (1 * 1024 *1024),
        Coalesce_Heuristic coalesce_heuristic = heuristic_percent_releasable(100)) noexcept;

    void* allocate(size_t bytes) override;

    void deallocate(void* ptr) override;

    void release() override;

    long getCurrentSize() const noexcept override;
    long getActualSize() const noexcept override;
    long getHighWatermark() const noexcept override;

    Platform getPlatform() noexcept override;

    /*!
     * \brief Get the number of bytes that may be released back to resource
     *
     * A memory pool has a set of blocks that have no allocations
     * against them.  If the size of the set is greater than one, then
     * the pool will have a number of bytes that may be released back to
     * the resource or coalesced into a larger block.
     *
     * \return The total number of bytes that are releasable
     */
    long getReleasableSize() const noexcept;

    /*!
     * \brief Get the number of memory blocks that the pools has
     *
     * \return The total number of blocks that are allocated by the pool
     */
    long getBlocksInPool() const noexcept;

    void coalesce() noexcept;

  private:
    DynamicSizePool* dpa;

    strategy::AllocationStrategy* m_allocator;
    Coalesce_Heuristic do_coalesce;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPool_HPP
