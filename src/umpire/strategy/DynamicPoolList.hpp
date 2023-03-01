//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DynamicPoolList_HPP
#define UMPIRE_DynamicPoolList_HPP

#include <functional>
#include <memory>
#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicSizePool.hpp"
#include "umpire/strategy/PoolCoalesceHeuristic.hpp"

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
class DynamicPoolList : public AllocationStrategy {
 public:
  /*!
   * \brief Coalescing Heuristic functions for Percent-Releasable and Blocks-Releasable. Both have
   * the option to reallocate to High Watermark instead of actual size of the pool (actual size is
   * currently the default).
   */
  static PoolCoalesceHeuristic<DynamicPoolList> percent_releasable(int percentage);
  static PoolCoalesceHeuristic<DynamicPoolList> percent_releasable_hwm(int percentage);
  static PoolCoalesceHeuristic<DynamicPoolList> blocks_releasable(std::size_t nblocks);
  static PoolCoalesceHeuristic<DynamicPoolList> blocks_releasable_hwm(std::size_t nblocks);

  static constexpr std::size_t s_default_first_block_size{512 * 1024 * 1024};
  static constexpr std::size_t s_default_next_block_size{1 * 1024 * 1024};
  static constexpr std::size_t s_default_alignment{16};

  /*!
   * \brief Construct a new DynamicPoolList.
   *
   * \param name Name of this instance of the DynamicPoolList.
   * \param id Id of this instance of the DynamicPoolList.
   * \param allocator Allocation resource that pool uses
   * \param first_minimum_pool_allocation_size Minimum size the pool initially
   * allocates \param next_minimum_pool_allocation_size The minimum size of all
   * future allocations. \param align_bytes Number of bytes with which to align
   * allocation sizes (power-of-2) \param do_heuristic Heuristic for when to
   * perform coalesce operation
   */
  DynamicPoolList(const std::string& name, int id, Allocator allocator,
                  const std::size_t first_minimum_pool_allocation_size = s_default_first_block_size,
                  const std::size_t next_minimum_pool_allocation_size = s_default_next_block_size,
                  const std::size_t alignment = s_default_alignment,
                  PoolCoalesceHeuristic<DynamicPoolList> should_coalesce = percent_releasable(100)) noexcept;

  DynamicPoolList(const DynamicPoolList&) = delete;

  void* allocate(size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;
  void release() override;

  std::size_t getReleasableBlocks() const noexcept;
  std::size_t getTotalBlocks() const noexcept;

  std::size_t getActualSize() const noexcept override;
  std::size_t getCurrentSize() const noexcept override;
  std::size_t getActualHighwaterMark() const noexcept;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept final override;

  bool tracksMemoryUse() const noexcept override;

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
  std::size_t getReleasableSize() const noexcept;

  /*!
   * \brief Get the number of memory blocks that the pool has
   *
   * \return The total number of blocks that are allocated by the pool
   */
  std::size_t getBlocksInPool() const noexcept;

  /*!
   * \brief Get the largest allocatable number of bytes from pool before
   * the pool will grow.
   *
   * return The largest number of bytes that may be allocated without
   * causing pool growth
   */
  std::size_t getLargestAvailableBlock() const noexcept;

  void coalesce() noexcept;

 private:
  strategy::AllocationStrategy* m_allocator;
  DynamicSizePool<> dpa;
  PoolCoalesceHeuristic<DynamicPoolList> m_should_coalesce;
};

std::ostream& operator<<(std::ostream& out, PoolCoalesceHeuristic<DynamicPoolList>&);

inline std::string to_string(PoolCoalesceHeuristic<DynamicPoolList>&)
{
  return "PoolCoalesceHeuristic<DynamicPoolList>";
}

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPoolList_HPP
