//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DynamicPoolMap_HPP
#define UMPIRE_DynamicPoolMap_HPP

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"

#include "umpire/util/MemoryMap.hpp"

#include <map>
#include <tuple>

namespace umpire {

class Allocator;

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
class DynamicPoolMap : public AllocationStrategy
{
  public:
    using Pointer = void*;

    /*!
     * \brief Callback heuristic to trigger coalesce of free blocks in pool.
     *
     * The registered heuristic callback function will be called immediately
     * after a deallocation() has completed from the pool.
     */
    using CoalesceHeuristic = std::function<bool (const strategy::DynamicPoolMap&)>;

    /*!
     * \brief Construct a new DynamicPoolMap.
     *
     * \param name Name of this instance of the DynamicPoolMap
     * \param id Unique identifier for this instance
     * \param initial_alloc_bytes Size the pool initially allocates
     * \param min_alloc_bytes The minimum size of all future allocations
     * \param coalesce_heuristic Heuristic callback function
     * \param align_bytes Number of bytes with which to align allocation sizes
     */
    DynamicPoolMap(
        const std::string& name,
        int id,
        Allocator allocator,
        const std::size_t initial_alloc_size = (512 * 1024 * 1024),
        const std::size_t min_alloc_size = (1 * 1024 * 1024),
        const std::size_t align_bytes = 16,
        CoalesceHeuristic coalesce_heuristic = heuristic_percent_releasable(100)) noexcept;

    /*!
     * \brief Destructs the DynamicPoolMap.
     */
    ~DynamicPoolMap();

    DynamicPoolMap(const DynamicPoolMap&) = delete;

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr) override;
    void release() override;

    std::size_t getActualSize() const noexcept override;

    Platform getPlatform() noexcept override;

    MemoryResourceTraits getTraits() const noexcept override;


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
     * \brief Get the largest allocatable number of bytes from pool before
     * the pool will grow.
     *
     * return The largest number of bytes that may be allocated without
     * causing pool growth
     */
    std::size_t getLargestAvailableBlock() noexcept;

    /*!
     * \brief Merge as many free records as possible, release all possible free
     * blocks, then reallocate a chunk to keep the actual size the same.
     */
    void coalesce();

  private:
    using SizeTuple = std::tuple<std::size_t, bool, std::size_t>;
    using AddressTuple = std::tuple<Pointer, bool, std::size_t>;
    using AddressMap = util::MemoryMap<SizeTuple>;
    using SizeMap = std::multimap<std::size_t, AddressTuple>;

    /*!
     * \brief Allocate from m_allocator.
     */
    void* allocateBlock(std::size_t bytes);

    /*!
     * \brief Deallocate from m_allocator.
     */
    void deallocateBlock(void* ptr, std::size_t bytes);

    /*!
     * \brief Insert a block to the used map.
     */
    void insertUsed(Pointer addr, std::size_t bytes, bool is_head,
                    std::size_t whole_bytes);

    /*!
     * \brief Insert a block to the free map.
     */
    void insertFree(Pointer addr, std::size_t bytes, bool is_head,
                    std::size_t whole_bytes);

    /*!
     * \brief Find a free block with (length <= bytes) as close to bytes in
     * length as possible.
     */
    SizeMap::const_iterator findFreeBlock(std::size_t bytes) const;

    /*!
     * \brief Merge all contiguous blocks in m_free_map.
     *
     * NOTE This method is rather expensive, but critical to avoid pool growth
     */
    void mergeFreeBlocks();

    /*!
     * \brief Release blocks from m_free_map that have is_head = true and return
     * the amount of memory released.
     */
    std::size_t releaseFreeBlocks();

    void do_coalesce();

    strategy::AllocationStrategy* m_allocator;
    const std::size_t m_initial_alloc_bytes;
    const std::size_t m_min_alloc_bytes;
    const std::size_t m_align_bytes;
    CoalesceHeuristic m_coalesce_heuristic;
    AddressMap m_used_map;
    SizeMap m_free_map;
    std::size_t m_actual_bytes;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPoolMap_HPP
