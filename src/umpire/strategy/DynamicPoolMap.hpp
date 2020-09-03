//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DynamicPoolMap_HPP
#define UMPIRE_DynamicPoolMap_HPP

#include <functional>
#include <map>
#include <tuple>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/mixins/AlignedAllocation.hpp"
#include "umpire/util/MemoryMap.hpp"

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
class DynamicPoolMap : public AllocationStrategy,
                       private mixins::AlignedAllocation {
 public:
  using Pointer = void*;

  using CoalesceHeuristic =
      std::function<bool(const strategy::DynamicPoolMap&)>;

  static CoalesceHeuristic percent_releasable(int percentage);

  /*!
   * \brief Construct a new DynamicPoolMap.
   *
   * \param name Name of this instance of the DynamicPoolMap
   * \param id Unique identifier for this instance
   * \param allocator Allocation resource that pool uses
   * \param first_minimum_pool_allocation_size Size the pool initially allocates
   * \param next_minimum_pool_allocation_size The minimum size of all future
   * allocations \param align_bytes Number of bytes with which to align
   * allocation sizes (power-of-2) \param should_coalesce Heuristic for when to
   * perform coalesce operation
   */
  DynamicPoolMap(
      const std::string& name, int id, Allocator allocator,
      const std::size_t first_minimum_pool_allocation_size = (512 * 1024 *
                                                              1024),
      const std::size_t min_alloc_size = (1 * 1024 * 1024),
      const std::size_t align_bytes = 16,
      CoalesceHeuristic should_coalesce = percent_releasable(100)) noexcept;

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
  void deallocateBlock(void* ptr, std::size_t size);

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

  AddressMap m_used_map{};
  SizeMap m_free_map{};

  CoalesceHeuristic m_should_coalesce;

  const std::size_t m_first_minimum_pool_allocation_size;
  const std::size_t m_next_minimum_pool_allocation_size;

  std::size_t m_actual_bytes{0};
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPoolMap_HPP
