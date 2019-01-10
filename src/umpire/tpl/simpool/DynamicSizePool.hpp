#ifndef _DYNAMICSIZEPOOL_HPP
#define _DYNAMICSIZEPOOL_HPP

#include <cstddef>
#include <cassert>
#include <string>
#include <iostream>
#include <sstream>

#include "umpire/tpl/simpool/StdAllocator.hpp"
#include "umpire/tpl/simpool/FixedSizePool.hpp"

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

template <class IA = StdAllocator>
class DynamicSizePool
{
protected:
  struct Block
  {
    char *data;
    std::size_t size;
    std::size_t blockSize;
    Block *next;
  };

  // Allocator for the underlying data
  typedef FixedSizePool<struct Block, IA, IA, (1<<6)> BlockPool;
  BlockPool blockPool;

  // Start of the nodes of used and free block lists
  struct Block *usedBlocks;
  struct Block *freeBlocks;

  // Total size allocated (bytes)
  std::size_t totalBytes;

  // Allocated size (bytes)
  std::size_t allocBytes;

  // Minimum size of initial allocation
  std::size_t minInitialBytes;

  // Minimum size for allocations
  std::size_t minBytes;

  // High water mark of allocations
  std::size_t highWaterMark;

  // Pointer to our allocator's allocation strategy
  std::shared_ptr<umpire::strategy::AllocationStrategy> allocator;

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
      data = allocator->allocate(sizeToAlloc);
    }
    catch (...) {
      UMPIRE_LOG(Error, 
          "\n\tMemory exhausted at allocation resource. "
          "Attempting to give blocks back.\n\t"
          << allocatedSize() << " Allocated to pool, "
          << numFreeBlocks() << " Free Blocks, "
          << numUsedBlocks() << " Used Blocks\n"
      );
      freeReleasedBlocks();
      UMPIRE_LOG(Error, 
          "\n\tMemory exhausted at allocation resource.  "
          "\n\tRetrying allocation operation: "
          << allocatedSize() << " Bytes still allocated to pool, "
          << numFreeBlocks() << " Partially Free Blocks, "
          << numUsedBlocks() << " Used Blocks\n"
      );
      try {
        data = allocator->allocate(sizeToAlloc);
        UMPIRE_LOG(Error, 
          "\n\tMemory successfully recovered at resource.  Allocation succeeded\n"
        );
      }
      catch (...) {
        UMPIRE_LOG(Error, 
          "\n\tUnable to allocate from resource even after giving back free blocks.\n"
          "\tThrowing to let application know we have no more memory: "
          << allocatedSize() << " Bytes still allocated to pool\n"
          << numFreeBlocks() << " Partially Free Blocks, "
          << numUsedBlocks() << " Used Blocks\n"
        );
        throw;
      }
    }

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
        totalBytes -= curr->blockSize;
        freed += curr->blockSize;
        allocator->deallocate(curr->data);

        if ( prev )   prev->next = curr->next;
        else          freeBlocks = curr->next;

        blockPool.deallocate(curr);
      }
      else {
        prev = curr;
      }
      curr = next;
    }

    return freed;
  }

  void coalesceFreeBlocks(std::size_t size) {
    UMPIRE_LOG(Debug, "Allocator " << this
                        << " coalescing to "
                        << size << " bytes from "
                        << numFreeBlocks() << " free blocks\n");
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

public:
  DynamicSizePool(
      std::shared_ptr<umpire::strategy::AllocationStrategy> strat,
      const std::size_t _minInitialBytes = (16 * 1024),
      const std::size_t _minBytes = 256
      )
    : blockPool(),
      usedBlocks(NULL),
      freeBlocks(NULL),
      totalBytes(0),
      allocBytes(0),
      minInitialBytes(_minInitialBytes),
      minBytes(_minBytes),
      highWaterMark(0),
      allocator(strat) { }

  ~DynamicSizePool() { freeAllBlocks(); }

  void *allocate(std::size_t size) {
    struct Block *best, *prev;
    size = alignmentAdjust(size);
    findUsableBlock(best, prev, size);

    // Allocate a block if needed
    if (!best) allocateBlock(best, prev, size);
    assert(best);

    // Split the free block
    splitBlock(best, prev, size);

    // Push node to the list of used nodes
    best->next = usedBlocks;
    usedBlocks = best;

    // Increment the allocated size
    allocBytes += size;

    if ( allocBytes > highWaterMark )
      highWaterMark = allocBytes;

    // Return the new pointer
    return usedBlocks->data;
  }

  void deallocate(void *ptr) {
    assert(ptr);

    // Find the associated block
    struct Block *curr = usedBlocks, *prev = NULL;
    for ( ; curr && curr->data != ptr; curr = curr->next ) {
      prev = curr;
    }
    if (!curr) return;

    // Remove from allocBytes
    allocBytes -= curr->size;

    // Release it
    releaseBlock(curr, prev);

    if ( allocBytes == 0 && numFreeBlocks() > 1 )
      coalesceFreeBlocks(highWaterMark);
  }

  std::size_t allocatedSize() const { return allocBytes; }

  std::size_t totalSize() const {
    return totalBytes + blockPool.totalSize();
  }

  std::size_t numFreeBlocks() const {
    std::size_t nb = 0;
    for (struct Block *temp = freeBlocks; temp; temp = temp->next) nb++;
    return nb;
  }

  std::size_t numUsedBlocks() const {
    std::size_t nb = 0;
    for (struct Block *temp = usedBlocks; temp; temp = temp->next) nb++;
    return nb;
  }

  void coalesce() {
    if ( numFreeBlocks() > 1 ) {
      std::size_t size_to_coalesce = freeReleasedBlocks();

      UMPIRE_LOG(Debug, "Attempting to coalesce "
                      << size_to_coalesce << " bytes");

      coalesceFreeBlocks(size_to_coalesce);
    }
  }

  void release()
  {
    freeReleasedBlocks();
  }
};

#endif
