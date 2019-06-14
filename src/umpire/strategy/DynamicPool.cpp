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

#include "umpire/strategy/DynamicPool.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"

#include <algorithm>

namespace umpire {
namespace strategy {


void DynamicPool::findUsableBlock(struct Block *&best, struct Block *&prev, std::size_t size) {
  best = prev = nullptr;
  for ( struct Block *iter = freeBlocks, *iterPrev = nullptr ; iter ; iter = iter->next ) {
    if ( iter->size >= size && (!best || iter->size < best->size) ) {
      best = iter;
      prev = iterPrev;
      if ( iter->size == size )
        break;    // Exact match, won't find a better one, look no further
    }
    iterPrev = iter;
  }
}

inline static std::size_t alignmentAdjust(const std::size_t size) {
  const std::size_t AlignmentBoundary = 16;
  return std::size_t (size + (AlignmentBoundary-1)) & ~(AlignmentBoundary-1);
}

void DynamicPool::allocateBlock(struct Block *&curr, struct Block *&prev, const std::size_t size) {
  std::size_t sizeToAlloc;

  if ( freeBlocks == nullptr && usedBlocks == nullptr )
    sizeToAlloc = std::max(size, minInitialBytes);
  else
    sizeToAlloc = std::max(size, minBytes);

  curr = prev = nullptr;
  void *data = nullptr;

  // Allocate data
  try {
    data = m_allocator->allocate(sizeToAlloc);
  }
  catch (...) {
    UMPIRE_LOG(Error,
               "\n\tMemory exhausted at allocation resource. "
               "Attempting to give blocks back.\n\t"
               << getCurrentSize() << " Allocated to pool, "
               << getFreeBlocks() << " Free Blocks, "
               << getInUseBlocks() << " Used Blocks\n"
      );
    freeReleasedBlocks();
    UMPIRE_LOG(Error,
               "\n\tMemory exhausted at allocation resource.  "
               "\n\tRetrying allocation operation: "
               << getCurrentSize() << " Bytes still allocated to pool, "
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
                 << getCurrentSize() << " Bytes still allocated to pool\n"
                 << getFreeBlocks() << " Partially Free Blocks, "
                 << getInUseBlocks() << " Used Blocks\n"
        );
      throw;
    }
  }

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

void DynamicPool::splitBlock(struct Block *&curr, struct Block *&prev, const std::size_t size) {
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

void DynamicPool::releaseBlock(struct Block *curr, struct Block *prev) {
  assert(curr != nullptr);

  if (prev) prev->next = curr->next;
  else usedBlocks = curr->next;

  // Find location to put this block in the freeBlocks list
  prev = nullptr;
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

std::size_t DynamicPool::freeReleasedBlocks() {
  // Release the unused blocks
  struct Block *curr = freeBlocks;
  struct Block *prev = nullptr;

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

  return freed;
}

void DynamicPool::coalesceFreeBlocks(std::size_t size) {
  UMPIRE_LOG(Debug, "Allocator " << this
             << " coalescing to "
             << size << " bytes from "
             << getFreeBlocks() << " free blocks\n");
  freeReleasedBlocks();
  void* ptr = allocate(size);
  deallocate(ptr);
}

void DynamicPool::freeAllBlocks() {
  // Release the used blocks
  while(usedBlocks) {
    releaseBlock(usedBlocks, nullptr);
  }

  freeReleasedBlocks();
  UMPIRE_ASSERT("Not all blocks were released properly" && freeBlocks == nullptr );
}

DynamicPool::DynamicPool(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::size_t min_initial_alloc_size,
    const std::size_t min_alloc_size,
    Coalesce_Heuristic coalesce_heuristic) noexcept :
  AllocationStrategy(name, id),
  blockPool(sizeof(struct Block)),
  usedBlocks(nullptr),
  freeBlocks(nullptr),
  totalBlocks(0),
  totalBytes(0),
  allocBytes(0),
  minInitialBytes(min_initial_alloc_size),
  minBytes(min_alloc_size),
  highWatermark(0),
  m_allocator(allocator.getAllocationStrategy()),
  do_coalesce(coalesce_heuristic)
{
}

DynamicPool::~DynamicPool() { freeAllBlocks(); }

void*
DynamicPool::allocate(size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");

  struct Block *best, *prev;
  bytes = alignmentAdjust(bytes);
  findUsableBlock(best, prev, bytes);

  // Allocate a block if needed
  if (!best) allocateBlock(best, prev, bytes);
  assert(best);

  // Split the free block
  splitBlock(best, prev, bytes);

  // Push node to the list of used nodes
  best->next = usedBlocks;
  usedBlocks = best;

  // Increment the allocated size
  allocBytes += bytes;

  if ( allocBytes > highWatermark )
    highWatermark = allocBytes;

  // Return the new pointer
  return usedBlocks->data;
}

void
DynamicPool::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  UMPIRE_ASSERT(ptr);

  // Find the associated block
  struct Block *curr = usedBlocks, *prev = nullptr;
  for ( ; curr && curr->data != ptr; curr = curr->next ) {
    prev = curr;
  }
  if (!curr) return;

  // Remove from allocBytes
  allocBytes -= curr->size;

  // Release it
  releaseBlock(curr, prev);

  if ( do_coalesce(*this) ) {
    UMPIRE_LOG(Debug, "Heuristic returned true, "
        "performing coalesce operation for " << this << "\n");
    coalesce();
  }
}

std::size_t DynamicPool::getCurrentSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << allocBytes);
  return allocBytes;
}

std::size_t DynamicPool::getActualSize() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << totalBytes);
  return totalBytes;
}

std::size_t DynamicPool::getHighWatermark() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << highWatermark);
  return highWatermark;
}

std::size_t DynamicPool::getBlocksInPool() const noexcept
{
  UMPIRE_LOG(Debug, "() returning " << totalBlocks);
  return totalBlocks;
}

std::size_t DynamicPool::getReleasableSize() const noexcept
{
  std::size_t nblocks = 0;
  std::size_t nbytes = 0;
  for (struct Block *temp = freeBlocks; temp; temp = temp->next) {
    if ( temp->size == temp->blockSize ) {
      nbytes += temp->blockSize;
      nblocks++;
    }
  }

  const std::size_t sparse_block_size = nblocks > 1 ? nbytes : 0;
  UMPIRE_LOG(Debug, "() returning " << sparse_block_size);
  return sparse_block_size;
}

std::size_t DynamicPool::getFreeBlocks() const
{
  std::size_t nb = 0;
  for (struct Block *temp = freeBlocks; temp; temp = temp->next)
    if ( temp->size == temp->blockSize )
      nb++;
  return nb;
}

std::size_t DynamicPool::getInUseBlocks() const
{
  std::size_t nb = 0;
  for (struct Block *temp = usedBlocks; temp; temp = temp->next) nb++;
  return nb;
}

Platform
DynamicPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void DynamicPool::coalesce() noexcept
{
  UMPIRE_REPLAY("\"event\": \"coalesce\", \"payload\": { \"allocator_name\": \"" << getName() << "\" }");
  if ( getFreeBlocks() > 1 ) {
    std::size_t size_to_coalesce = freeReleasedBlocks();

    UMPIRE_LOG(Debug, "Attempting to coalesce "
               << size_to_coalesce << " bytes");

    coalesceFreeBlocks(size_to_coalesce);
  }
}

void
DynamicPool::release()
{
  freeReleasedBlocks();
}

} // end of namespace strategy
} // end of namespace umpire
