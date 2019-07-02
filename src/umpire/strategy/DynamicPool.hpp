//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DynamicPool_HPP
#define UMPIRE_DynamicPool_HPP

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"
#include "umpire/util/MemoryMap.hpp"

#include "umpire/Allocator.hpp"

#include <map>
#include <utility>

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
class DynamicPool : public AllocationStrategy
{
  public:
    using Pointer = void*;

    /*!
     * \brief Callback heuristic to trigger coalesce of free blocks in pool.
     *
     * The registered heuristic callback function will be called immediately
     * after a deallocation() has completed from the pool.
     */
    using CoalesceHeuristic = std::function<bool (const strategy::DynamicPool&)>;

    /*!
     * \brief Construct a new DynamicPool.
     *
     * \param name Name of this instance of the DynamicPool
     * \param id Unique identifier for this instance
     * \param initial_alloc_bytes Size the pool initially allocates
     * \param min_alloc_bytes The minimum size of all future allocations
     * \param coalesce_heuristic Heuristic callback function
     * \param align_bytes Number of bytes with which to align allocation sizes
     */
    DynamicPool(
        const std::string& name,
        int id,
        Allocator allocator,
        const std::size_t initial_alloc_size = (512 * 1024 * 1024),
        const std::size_t min_alloc_size = (1 * 1024 * 1024),
        CoalesceHeuristic coalesce_heuristic = heuristic_percent_releasable(100),
        const int align_bytes = 16) noexcept;

    /*!
     * \brief Destructs the DynamicPool.
     */
    ~DynamicPool();

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr) override;
    void release() override;

    std::size_t getCurrentSize() const noexcept override;
    std::size_t getActualSize() const noexcept override;
    std::size_t getHighWatermark() const noexcept override;

    Platform getPlatform() noexcept override;

    /*!
     * \brief Returns the number of bytes of unallocated data held by this pool
     * that could be immediately released back to the resource.
     *
     * A memory pool has a set of blocks that are not leased out to the
     * application as allocations. Allocations from the resource begin as a
     * single chunk, but these could be split, and only the first chunk can be
     * deallocated back to the resource immediately.
     *
     * \return The total number of bytes that are immediately releasable.
     */
    std::size_t getReleasableSize() const noexcept;

    /*!
     * \brief Return the number of free memory blocks that the pools holds.
     */
    std::size_t getFreeBlocks() const noexcept;

    /*!
     * \brief Return the number of used memory blocks that the pools holds.
     */
    std::size_t getInUseBlocks() const noexcept;

    /*!
     * \brief Return the number of memory blocks -- both leased to application
     * and internal free memory -- that the pool holds.
     */
    std::size_t getBlocksInPool() const noexcept;

    /*!
     * \brief Merge as many free records as possible, release all possible free
     * blocks, then reallocate a chunk to keep the actual size the same.
     */
    void coalesce();

  private:
    using SizePair = std::pair<std::size_t, bool>;
    using AddressPair = std::pair<Pointer, bool>;
    using AddressMap = util::MemoryMap<SizePair>;
    using SizeMap = std::multimap<std::size_t, AddressPair>;

    // Insert a block to the used map
    void insertUsed(Pointer addr, std::size_t bytes, bool is_head);

    // Insert a block to the free map
    void insertFree(Pointer addr, std::size_t bytes, bool is_head);

    // find a free block with length <= bytes as close to bytes in length as
    // possible.
    SizeMap::const_iterator findFreeBlock(std::size_t bytes) const;

    void doCoalesce();
    std::size_t doRelease();

    strategy::AllocationStrategy* m_allocator;
    const std::size_t m_min_alloc_bytes;
    const int m_align_bytes;
    CoalesceHeuristic m_coalesce_heuristic;
    AddressMap m_used_map;
    SizeMap m_free_map;
    std::size_t m_curr_bytes;
    std::size_t m_actual_bytes;
    std::size_t m_highwatermark;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPool_HPP
