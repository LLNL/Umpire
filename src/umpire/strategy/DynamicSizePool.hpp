//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef _DYNAMICSIZEPOOL_HPP
#define _DYNAMICSIZEPOOL_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/FixedSizePool.hpp"
#include "umpire/strategy/StdAllocator.hpp"
#include "umpire/strategy/mixins/AlignedAllocation.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/memory_sanitizers.hpp"

template <class IA = StdAllocator>
class DynamicSizePool : private umpire::strategy::mixins::AlignedAllocation {
 protected:
  struct Block {
    char *data;
    std::size_t size;
    std::size_t blockSize;
    Block *next;
  };

  // Allocator for the underlying data
  typedef FixedSizePool<struct Block, IA, IA, (1 << 6)> BlockPool;
  BlockPool blockPool{};

  // Start of the nodes of used and free block lists
  struct Block *usedBlocks{nullptr};
  struct Block *freeBlocks{nullptr};

  // Total size allocated (bytes)
  std::size_t m_actual_bytes{0};
  std::size_t m_current_size{0};

  // Minimum size of initial allocation
  std::size_t m_first_minimum_pool_allocation_size;

  // Minimum size for allocations
  std::size_t m_next_minimum_pool_allocation_size;

  std::size_t m_releasable_blocks{0};
  std::size_t m_total_blocks{0};

  bool m_is_destructing{false};

  // Search the list of free blocks and return a usable one if that exists, else
  // NULL
  void findUsableBlock(struct Block *&best, struct Block *&prev,
                       std::size_t size)
  {
    best = prev = NULL;
    for (struct Block *iter = freeBlocks, *iterPrev = NULL; iter;
         iter = iter->next) {
      if (iter->size >= size && (!best || iter->size < best->size)) {
        best = iter;
        prev = iterPrev;
        if (iter->size == size)
          break; // Exact match, won't find a better one, look no further
      }
      iterPrev = iter;
    }
  }

  // Allocate a new block and add it to the list of free blocks
  void allocateBlock(struct Block *&curr, struct Block *&prev, std::size_t size)
  {
    if (freeBlocks == NULL && usedBlocks == NULL)
      size = std::max(size, m_first_minimum_pool_allocation_size);
    else
      size = std::max(size, m_next_minimum_pool_allocation_size);

    UMPIRE_LOG(Debug, "Allocating new chunk of size " << size);

    curr = nullptr;
    prev = nullptr;
    void *data{nullptr};

    try {
#if defined(UMPIRE_ENABLE_BACKTRACE)
      {
        umpire::util::backtrace bt;
        umpire::util::backtracer<>::get_backtrace(bt);
        UMPIRE_LOG(Info,
                   "actual_size:" << (m_actual_bytes + size)
                                  << " (prev: " << m_actual_bytes << ") "
                                  << umpire::util::backtracer<>::print(bt));
      }
#endif
      data = aligned_allocate(size); // Will POISON
    } catch (...) {
      UMPIRE_LOG(Error,
                 "Caught error allocating new chunk, giving up free chunks and "
                 "retrying...");
      freeReleasedBlocks();
      try {
        data = aligned_allocate(size); // Will POISON
        UMPIRE_LOG(Debug, "memory reclaimed, chunk successfully allocated.");
      } catch (...) {
        UMPIRE_LOG(Error, "recovery failed.");
        throw;
      }
    }

    m_actual_bytes += size;
    m_releasable_blocks++;
    m_total_blocks++;

    // Allocate the block
    curr = (struct Block *)blockPool.allocate();
    assert("Failed to allocate block for freeBlock List" && curr);

    // Find next and prev such that next->data is still smaller than data (keep
    // ordered)
    struct Block *next;
    for (next = freeBlocks; next && next->data < data; next = next->next)
      prev = next;

    // Insert
    curr->data = static_cast<char *>(data);
    curr->size = size;
    curr->blockSize = size;
    curr->next = next;

    // Insert
    if (prev)
      prev->next = curr;
    else
      freeBlocks = curr;
  }

  void splitBlock(struct Block *&curr, struct Block *&prev,
                  const std::size_t size)
  {
    struct Block *next;

    if (curr->size == curr->blockSize)
      m_releasable_blocks--;

    if (curr->size == size) {
      // Keep it
      next = curr->next;
    } else {
      // Split the block
      std::size_t remaining = curr->size - size;
      struct Block *newBlock = (struct Block *)blockPool.allocate();
      if (!newBlock)
        return;
      newBlock->data = curr->data + size;
      newBlock->size = remaining;
      newBlock->blockSize = 0;
      newBlock->next = curr->next;
      next = newBlock;
      curr->size = size;
    }

    if (prev)
      prev->next = next;
    else
      freeBlocks = next;
  }

  void releaseBlock(struct Block *curr, struct Block *prev)
  {
    assert(curr != NULL);

    if (prev)
      prev->next = curr->next;
    else
      usedBlocks = curr->next;

    // Find location to put this block in the freeBlocks list
    prev = NULL;
    for (struct Block *temp = freeBlocks; temp && (temp->data < curr->data);
         temp = temp->next)
      prev = temp;

    // Keep track of the successor
    struct Block *next = prev ? prev->next : freeBlocks;

    // Check if prev and curr can be merged
    if (prev && prev->data + prev->size == curr->data && !curr->blockSize) {
      prev->size = prev->size + curr->size;
      blockPool.deallocate(curr); // keep data
      curr = prev;
    } else if (prev) {
      prev->next = curr;
    } else {
      freeBlocks = curr;
    }

    // Check if curr and next can be merged
    if (next && curr->data + curr->size == next->data && !next->blockSize) {
      curr->size = curr->size + next->size;
      curr->next = next->next;
      blockPool.deallocate(next); // keep data
    } else {
      curr->next = next;
    }

    if (curr->size == curr->blockSize)
      m_releasable_blocks++;
  }

  std::size_t freeReleasedBlocks()
  {
    // Release the unused blocks
    struct Block *curr = freeBlocks;
    struct Block *prev = NULL;

    std::size_t freed = 0;

    while (curr) {
      struct Block *next = curr->next;
      // The free block list may contain partially released released blocks.
      // Make sure to only free blocks that are completely released.
      //
      if (curr->size == curr->blockSize) {
        UMPIRE_LOG(Debug, "Releasing " << curr->size << " size chunk @ "
                                       << static_cast<void *>(curr->data));

        m_actual_bytes -= curr->size;
        m_releasable_blocks--;
        m_total_blocks--;

        freed += curr->size;
        try {
          aligned_deallocate(curr->data);
        } catch (...) {
          if (m_is_destructing) {
            //
            // Ignore error in case the underlying vendor API has already
            // shutdown
            //
            UMPIRE_LOG(Error, "Pool is destructing, Exception Ignored");
          } else {
            throw;
          }
        }

        if (prev)
          prev->next = curr->next;
        else
          freeBlocks = curr->next;

        blockPool.deallocate(curr);
      } else {
        prev = curr;
      }
      curr = next;
    }

#if defined(UMPIRE_ENABLE_BACKTRACE)
    if (freed > 0) {
      umpire::util::backtrace bt;
      umpire::util::backtracer<>::get_backtrace(bt);
      UMPIRE_LOG(Info, "actual_size:" << (m_actual_bytes) << " (prev: "
                                      << (m_actual_bytes + freed) << ") "
                                      << umpire::util::backtracer<>::print(bt));
    }
#endif

    return freed;
  }

  void coalesceFreeBlocks(std::size_t size)
  {
    UMPIRE_LOG(Debug, "Allocator " << this << " coalescing to " << size
                                   << " bytes from " << getFreeBlocks()
                                   << " free blocks\n");
    freeReleasedBlocks();
    void *ptr = allocate(size);
    deallocate(ptr);
  }

 public:
  DynamicSizePool(umpire::strategy::AllocationStrategy *strat,
                  const std::size_t first_minimum_pool_allocation_size = (16 *
                                                                          1024),
                  const std::size_t next_minimum_pool_allocation_size = 256,
                  const std::size_t alignment = 16)
      : umpire::strategy::mixins::AlignedAllocation{alignment, strat},
        m_first_minimum_pool_allocation_size{first_minimum_pool_allocation_size},
        m_next_minimum_pool_allocation_size{next_minimum_pool_allocation_size}
  {
    UMPIRE_LOG(Debug, " ( "
                          << ", allocator=\"" << strat->getName() << "\""
                          << ", first_minimum_pool_allocation_size="
                          << m_first_minimum_pool_allocation_size
                          << ", next_minimum_pool_allocation_size="
                          << m_next_minimum_pool_allocation_size
                          << ", alignment=" << alignment << " )");
  }

  DynamicSizePool(const DynamicSizePool &) = delete;

  ~DynamicSizePool()
  {
    UMPIRE_LOG(Debug, "Releasing free blocks to device");
    m_is_destructing = true;
    freeReleasedBlocks();
  }

  void *allocate(std::size_t bytes)
  {
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");
    const std::size_t rounded_bytes{aligned_round_up(bytes)};
    struct Block *best{nullptr}, *prev{nullptr};

    findUsableBlock(best, prev, rounded_bytes);

    // Allocate a block if needed
    if (!best) {
      allocateBlock(best, prev, rounded_bytes);
    }

    // Split the free block
    splitBlock(best, prev, rounded_bytes);

    // Push node to the list of used nodes
    best->next = usedBlocks;
    usedBlocks = best;

    m_current_size += rounded_bytes;
    UMPIRE_UNPOISON_MEMORY_REGION(m_allocator, usedBlocks->data, bytes);

    // Return the new pointer
    return usedBlocks->data;
  }

  void deallocate(void *ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

    // Find the associated block
    struct Block *curr = usedBlocks, *prev = NULL;
    for (; curr && curr->data != ptr; curr = curr->next) {
      prev = curr;
    }
    if (!curr)
      return;

    m_current_size -= curr->size;
    UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, curr->size);

    UMPIRE_LOG(Debug, "Deallocating data held by " << curr);
    // Release it
    releaseBlock(curr, prev);
  }

  void release()
  {
    UMPIRE_LOG(Debug, "()");
    freeReleasedBlocks();
  }

  std::size_t getReleasableBlocks() const noexcept
  {
    return m_releasable_blocks;
  }

  std::size_t getTotalBlocks() const noexcept
  {
    return m_total_blocks;
  }

  std::size_t getActualSize() const
  {
    return m_actual_bytes;
  }

  std::size_t getCurrentSize() const
  {
    return m_current_size;
  }

  std::size_t getBlocksInPool() const
  {
    std::size_t total_blocks{0};
    struct Block *curr{nullptr};

    for (curr = usedBlocks; curr; curr = curr->next) {
      total_blocks += 1;
    }
    for (curr = freeBlocks; curr; curr = curr->next) {
      total_blocks += 1;
    }

    return total_blocks;
  }

  std::size_t getLargestAvailableBlock() const
  {
    std::size_t largest_block{0};
    for (struct Block *temp = freeBlocks; temp; temp = temp->next)
      if (temp->size > largest_block)
        largest_block = temp->size;
    return largest_block;
  }

  std::size_t getReleasableSize() const
  {
    std::size_t nblocks = 0;
    std::size_t nbytes = 0;
    for (struct Block *temp = freeBlocks; temp; temp = temp->next) {
      if (temp->size == temp->blockSize) {
        nbytes += temp->blockSize;
        nblocks++;
      }
    }
    return nblocks > 1 ? nbytes : 0;
  }

  std::size_t getFreeBlocks() const
  {
    std::size_t nb = 0;
    for (struct Block *temp = freeBlocks; temp; temp = temp->next)
      if (temp->size == temp->blockSize)
        nb++;
    return nb;
  }

  std::size_t getInUseBlocks() const
  {
    std::size_t nb = 0;
    for (struct Block *temp = usedBlocks; temp; temp = temp->next)
      nb++;
    return nb;
  }

  void coalesce()
  {
    if (getFreeBlocks() > 1) {
      std::size_t size_to_coalesce = freeReleasedBlocks();

      UMPIRE_LOG(Debug, "coalescing " << size_to_coalesce << " bytes.");
      coalesceFreeBlocks(size_to_coalesce);
    }
  }
};

#endif
