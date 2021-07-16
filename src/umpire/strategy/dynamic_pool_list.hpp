//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/strategy/allocation_strategy.hpp"

#include <memory>
#include <vector>
#include <functional>

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
template<typename Memory=memory, bool Tracking=true>
class dynamic_pool_list :
  public allocation_strategy
{
  private:
  struct Block;

  public:
    /*!
     * \brief Callback Heuristic to trigger coalesce of free blocks in pool.
     *
     * The registered heuristic callback function will be called immediately
     * after a deallocation() has completed from the pool.
     */
    using CoalesceHeuristic = std::function<bool( const strategy::dynamic_pool_list<Memory, Tracking>& )>;
    static CoalesceHeuristic percent_releasable( int percentage )
    {
      if ( percentage < 0 || percentage > 100 ) {
        UMPIRE_ERROR("Invalid percentage of " << percentage 
            << ", percentage must be an integer between 0 and 100");
      }

      if ( percentage == 0 ) {
        return [=] (const strategy::dynamic_pool_list<Memory, Tracking>& UMPIRE_UNUSED_ARG(pool)) {
            return false;
        };
      } else if ( percentage == 100 ) {
        return [=] (const strategy::dynamic_pool_list<Memory, Tracking>& pool) {
            return (pool.get_current_size() == 0 && pool.getReleasableSize() > 0);
        };
      }

      float f = (float)((float)percentage / (float)100.0);
      return [=] (const strategy::dynamic_pool_list<Memory, Tracking>& pool) {
        // Calculate threshold in bytes from the percentage
        const std::size_t threshold = static_cast<std::size_t>(f * pool.get_actual_size());
        return (pool.getReleasableSize() >= threshold);
      };
    }

    /*!
     * \brief Construct a new DynamicPoolList.
     *
     * \param name Name of this instance of the DynamicPoolList.
     * \param id Id of this instance of the DynamicPoolList.
     * \param min_initial_alloc_size The minimum size of the first allocation
     *                               the pool will make.
     * \param min_alloc_size The minimum size of all future allocations.
     * \param coalesce_heuristic Heuristic callback function.
     */
    dynamic_pool_list(
        const std::string& name,
        Memory* allocator,
        const std::size_t min_initial_alloc_size = (512 * 1024 * 1024),
        const std::size_t min_alloc_size = (1 * 1024 *1024),
        CoalesceHeuristic coalesce_heuristic = percent_releasable(100)) noexcept :
        allocation_strategy{name},
        blockPool{sizeof(Block), (1<<6)},
        minInitialBytes{min_initial_alloc_size},
        minBytes{min_alloc_size},
        m_allocator{allocator},
        do_coalesce{coalesce_heuristic}
    {
    }

    void* allocate(size_t bytes) override
    {
      struct Block *best, *prev;
      bytes = alignmentAdjust(bytes);
      findUsableBlock(best, prev, bytes);

      // Allocate a block if needed
      if (!best)
        allocateBlock(best, prev, bytes);
      assert(best);

      // Split the free block
      splitBlock(best, prev, bytes);

      // Push node to the list of used nodes
      best->next = usedBlocks;
      usedBlocks = best;

      // Increment the allocated bytes
      allocBytes += bytes;

      if (allocBytes > highWatermark)
        highWatermark = allocBytes;

      UMPIRE_UNPOISON_MEMORY_REGION(m_allocator, usedBlocks->data, bytes);

      // Return the new pointer
      return usedBlocks->data;
    }

    void deallocate(void* ptr) override
    {
      assert(ptr);

      // Find the associated block
      struct Block *curr = usedBlocks, *prev = NULL;
      for ( ; curr && curr->data != ptr; curr = curr->next ) {
        prev = curr;
      }
      if (!curr) return;

      // Remove from allocBytes
      allocBytes -= curr->size;

      UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, curr->size);

      // Release it
      releaseBlock(curr, prev);

      if ( do_coalesce(*this) ) {
        UMPIRE_LOG(Debug, "Heuristic returned true, "
            "performing coalesce operation for " << this << "\n");
        coalesce();
      }
    }

    void release() // override
    {
      freeReleasedBlocks();
    }

    std::size_t get_actual_size() const noexcept override {
      return totalBytes;
    }

    camp::resources::Platform get_platform() noexcept override
    {
      return m_allocator->get_platform();
    }

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
    std::size_t getReleasableSize() const noexcept
    {
      std::size_t nblocks = 0;
      std::size_t nbytes = 0;
      for (struct Block *temp = freeBlocks; temp; temp = temp->next) {
        if ( temp->size == temp->blockSize ) {
          nbytes += temp->blockSize;
          nblocks++;
        }
      }
      return nblocks > 1 ? nbytes : 0;
    }

    /*!
     * \brief Get the number of memory blocks that the pool has
     *
     * \return The total number of blocks that are allocated by the pool
     */
    std::size_t getBlocksInPool() const noexcept
    {
      return totalBlocks;
    }

    /*!
     * \brief Get the largest allocatable number of bytes from pool before
     * the pool will grow.
     *
     * return The largest number of bytes that may be allocated without 
     * causing pool growth
     */
    std::size_t getLargestAvailableBlock() const noexcept
    {
      std::size_t largest_block{0};
      for (struct Block *temp = freeBlocks; temp; temp = temp->next)
        if ( temp->size > largest_block )
          largest_block = temp->size;
      return largest_block;
    }

    void coalesce() noexcept
    {
      UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \"" << get_name() << "\" }");

      if ( getFreeBlocks() > 1 ) {
        std::size_t size_to_coalesce = freeReleasedBlocks();

        UMPIRE_LOG(Debug, "Attempting to coalesce "
                        << size_to_coalesce << " bytes");

        coalesceFreeBlocks(size_to_coalesce);
      }
    }

  private:
  // Search the list of free blocks and return a usable one if that exists, else NULL
  void findUsableBlock(struct Block *&best, struct Block *&prev, std::size_t size) {
    best = prev = NULL;
    for ( struct Block *iter = freeBlocks, *iterPrev = NULL ; iter ; iter = iter->next ) {
      if ( iter->size >= size && (!best || iter->size < best->size) ) {
        best = iter;
        prev = iterPrev;
        if ( iter->size == size )
          break;    // Exact match, won't find a better one, look no further
      }
      iterPrev = iter;
    }
  }

  inline std::size_t alignmentAdjust(const std::size_t size) {
    const std::size_t AlignmentBoundary = 16;
    return std::size_t (size + (AlignmentBoundary-1)) & ~(AlignmentBoundary-1);
  }

  // Allocate a new block and add it to the list of free blocks
  void allocateBlock(struct Block *&curr, struct Block *&prev, const std::size_t size) {
    std::size_t sizeToAlloc;

    if ( freeBlocks == NULL && usedBlocks == NULL )
      sizeToAlloc = std::max(size, minInitialBytes);
    else
      sizeToAlloc = std::max(size, minBytes);

    curr = prev = NULL;
    void *data = NULL;

    // Allocate data
    try {
#if defined(UMPIRE_ENABLE_BACKTRACE)
      {
        umpire::util::backtrace bt{};
        umpire::util::backtracer<>::get_backtrace(bt);
        UMPIRE_LOG(Info, "actual_size:" << (totalBytes+sizeToAlloc) 
          << " (prev: " << totalBytes << ") " 
          << umpire::util::backtracer<>::print(bt));
      }
#endif
      data = m_allocator->allocate(sizeToAlloc);
    }
    catch (...) {
      UMPIRE_LOG(Error, 
          "\n\tMemory exhausted at allocation resource. "
          "Attempting to give blocks back.\n\t"
          << get_current_size() << " Allocated to pool, "
          << getFreeBlocks() << " Free Blocks, "
          << getInUseBlocks() << " Used Blocks\n"
      );
      freeReleasedBlocks();
      UMPIRE_LOG(Error, 
          "\n\tMemory exhausted at allocation resource.  "
          "\n\tRetrying allocation operation: "
          << get_current_size() << " Bytes still allocated to pool, "
          << getFreeBlocks() << " Free Blocks, "
          << getInUseBlocks() << " Used Blocks\n"
      );
      try {
        data = m_allocator->allocate(sizeToAlloc);
        UMPIRE_LOG(Error, 
          "\n\tMemory successfully recovered at resource.  Allocation succeeded\n"
        );
      }
      catch (...) {
        UMPIRE_LOG(Error, 
          "\n\tUnable to allocate from resource even after giving back free blocks.\n"
          "\tThrowing to let application know we have no more memory: "
          << get_current_size() << " Bytes still allocated to pool\n"
          << getFreeBlocks() << " Partially Free Blocks, "
          << getInUseBlocks() << " Used Blocks\n"
        );
        throw;
      }
    }

    UMPIRE_POISON_MEMORY_REGION(m_allocator, data, sizeToAlloc);

    totalBlocks += 1;
    totalBytes += sizeToAlloc;

    // Allocate the block
    curr = (struct Block *) blockPool.allocate();
    assert("Failed to allocate block for freeBlock List" && curr);

    // Find next and prev such that next->data is still smaller than data (keep ordered)
    struct Block *next;
    for ( next = freeBlocks; next && next->data < data; next = next->next )
      prev = next;

    // Insert
    curr->data = static_cast<char *>(data);
    curr->size = sizeToAlloc;
    curr->blockSize = sizeToAlloc;
    curr->next = next;

    // Insert
    if (prev) prev->next = curr;
    else freeBlocks = curr;
  }

  void splitBlock(struct Block *&curr, struct Block *&prev, const std::size_t size) {
    struct Block *next;

    if ( curr->size == size ) {
      // Keep it
      next = curr->next;
    }
    else {
      // Split the block
      std::size_t remaining = curr->size - size;
      struct Block *newBlock = (struct Block *) blockPool.allocate();
      if (!newBlock) return;
      newBlock->data = curr->data + size;
      newBlock->size = remaining;
      newBlock->blockSize = 0;
      newBlock->next = curr->next;
      next = newBlock;
      curr->size = size;
    }

    if (prev) prev->next = next;
    else freeBlocks = next;
  }

  void releaseBlock(struct Block *curr, struct Block *prev) {
    assert(curr != NULL);

    if (prev) prev->next = curr->next;
    else usedBlocks = curr->next;

    // Find location to put this block in the freeBlocks list
    prev = NULL;
    for ( struct Block *temp = freeBlocks ; temp && temp->data < curr->data ; temp = temp->next )
      prev = temp;

    // Keep track of the successor
    struct Block *next = prev ? prev->next : freeBlocks;

    // Check if prev and curr can be merged
    if ( prev && prev->data + prev->size == curr->data && !curr->blockSize ) {
      prev->size = prev->size + curr->size;
      blockPool.deallocate(curr); // keep data
      curr = prev;
    }
    else if (prev) {
      prev->next = curr;
    }
    else {
      freeBlocks = curr;
    }

    // Check if curr and next can be merged
    if ( next && curr->data + curr->size == next->data && !next->blockSize ) {
      curr->size = curr->size + next->size;
      curr->next = next->next;
      blockPool.deallocate(next); // keep data
    }
    else {
      curr->next = next;
    }
  }

  std::size_t freeReleasedBlocks() {
    // Release the unused blocks
    struct Block *curr = freeBlocks;
    struct Block *prev = NULL;

    std::size_t freed = 0;

    while ( curr ) {
      struct Block *next = curr->next;
      // The free block list may contain partially released released blocks.
      // Make sure to only free blocks that are completely released.
      //
      if ( curr->size == curr->blockSize ) {
        totalBlocks -= 1;
        totalBytes -= curr->blockSize;
        freed += curr->blockSize;
        m_allocator->deallocate(curr->data);

        if ( prev )   prev->next = curr->next;
        else          freeBlocks = curr->next;

        blockPool.deallocate(curr);
      }
      else {
        prev = curr;
      }
      curr = next;
    }

#if defined(UMPIRE_ENABLE_BACKTRACE)
    if (freed > 0) {
      umpire::util::backtrace bt{};
      umpire::util::backtracer<>::get_backtrace(bt);
      UMPIRE_LOG(Info, "actual_size:" << (totalBytes) 
        << " (prev: " << (totalBytes+freed) 
        << ") " << umpire::util::backtracer<>::print(bt));
    }
#endif

    return freed;
  }

  void coalesceFreeBlocks(std::size_t size) {
    UMPIRE_LOG(Debug, "Allocator " << this
                        << " coalescing to "
                        << size << " bytes from "
                        << getFreeBlocks() << " free blocks\n");
    freeReleasedBlocks();
    void* ptr = allocate(size);
    deallocate(ptr);
  }

  void freeAllBlocks() {
    // Release the used blocks
    while(usedBlocks) {
      releaseBlock(usedBlocks, NULL);
    }

    freeReleasedBlocks();
    assert( "Not all blocks were released properly" && freeBlocks == NULL );
  }

  std::size_t getFreeBlocks() const {
    std::size_t nb = 0;
    for (struct Block *temp = freeBlocks; temp; temp = temp->next)
      if ( temp->size == temp->blockSize )
        nb++;
    return nb;
  }

  std::size_t getInUseBlocks() const {
    std::size_t nb = 0;
    for (struct Block *temp = usedBlocks; temp; temp = temp->next) nb++;
    return nb;
  }

    struct Block {
      char *data;
      std::size_t size;
      std::size_t blockSize;
      Block *next;
    };

    // Allocator for the underlying data
    // using BlockPool = FixedSizePool<struct Block, IA, IA, (1<<6)>;
    using BlockPool = detail::fixed_malloc_pool;
    BlockPool blockPool;

    // Start of the nodes of used and free block lists
    struct Block *usedBlocks{nullptr};
    struct Block *freeBlocks{nullptr};

  // Total blocks in the pool
  std::size_t totalBlocks{0};

  // Total size allocated (bytes)
  std::size_t totalBytes{0};

  // Allocated size (bytes)
  std::size_t allocBytes{0};

  // Minimum size of initial allocation
  std::size_t minInitialBytes{0};

  // Minimum size for allocations
  std::size_t minBytes{0};

  // High water mark of allocations
  std::size_t highWatermark{0};

  Memory* m_allocator{nullptr};
  CoalesceHeuristic do_coalesce;
};

} // end of namespace strategy
} // end namespace umpire
