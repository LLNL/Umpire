#ifndef _DYNAMICSIZEPOOL_HPP
#define _DYNAMICSIZEPOOL_HPP

#include <algorithm>
#include <cstddef>
#include <cassert>
#include <string>
#include <iostream>
#include <sstream>

#include "umpire/util/FixedMallocPool.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

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
  umpire::util::FixedMallocPool blockPool;

  // Start of the nodes of used and free block lists
  struct Block *usedBlocks;1
  struct Block *freeBlocks;

  // Total blocks in the pool
  std::size_t totalBlocks;

  // Total size allocated (bytes)
  std::size_t totalBytes;

  // Allocated size (bytes)
  std::size_t allocBytes;

  // Minimum size of initial allocation
  std::size_t minInitialBytes;

  // Minimum size for allocations
  std::size_t minBytes;

  // High water mark of allocations
  std::size_t highWatermark;

  // Pointer to our allocator's allocation strategy
  umpire::strategy::AllocationStrategy* allocator;

  // Search the list of free blocks and return a usable one if that exists, else NULL
  void findUsableBlock(struct Block *&best, struct Block *&prev, std::size_t size);

  // Allocate a new block and add it to the list of free blocks
  void allocateBlock(struct Block *&curr, struct Block *&prev, const std::size_t size);

  void splitBlock(struct Block *&curr, struct Block *&prev, const std::size_t size);

  void releaseBlock(struct Block *curr, struct Block *prev);

  std::size_t freeReleasedBlocks();

  void coalesceFreeBlocks(std::size_t size);

  void freeAllBlocks();

public:
  DynamicSizePool(
      umpire::strategy::AllocationStrategy* strat,
      const std::size_t _minInitialBytes = (16 * 1024),
      const std::size_t _minBytes = 256);
  ~DynamicSizePool();

  void *allocate(std::size_t size);
  void deallocate(void *ptr);

  std::size_t getCurrentSize() const;
  std::size_t getActualSize() const;
  std::size_t getHighWatermark() const;
  std::size_t getBlocksInPool() const;
  std::size_t getReleasableSize() const;
  std::size_t getFreeBlocks() const;
  std::size_t getInUseBlocks() const;

  void coalesce();
  void release();
};

#endif // _DYNAMICSIZEPOOL_HPP
