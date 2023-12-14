// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "camp/camp.hpp"
#include "camp/resource.hpp"
#include "cuda_runtime_api.h"

#include "umpire/Allocator.hpp"
#include "umpire/strategy/PoolCoalesceHeuristic.hpp"
#include "umpire/strategy/StreamAwareQuickPool.hpp"
#include "umpire/strategy/mixins/AlignedAllocation.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/memory_sanitizers.hpp"

namespace umpire {
namespace strategy {

StreamAwareQuickPool::StreamAwareQuickPool(const std::string& name, int id, Allocator allocator,
                     const std::size_t first_minimum_pool_allocation_size,
                     const std::size_t next_minimum_pool_allocation_size, std::size_t alignment,
                     PoolCoalesceHeuristic<StreamAwareQuickPool> should_coalesce) noexcept
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "StreamAwareQuickPool"},
      mixins::AlignedAllocation{alignment, allocator.getAllocationStrategy()},
      m_should_coalesce{should_coalesce},
      m_first_minimum_pool_allocation_size{first_minimum_pool_allocation_size},
      m_next_minimum_pool_allocation_size{next_minimum_pool_allocation_size}
{
  UMPIRE_LOG(Debug, " ( "
                        << "name=\"" << name << "\""
                        << ", id=" << id << ", allocator=\"" << allocator.getName() << "\""
                        << ", first_minimum_pool_allocation_size=" << m_first_minimum_pool_allocation_size
                        << ", next_minimum_pool_allocation_size=" << m_next_minimum_pool_allocation_size
                        << ", alignment=" << alignment << " )");
}

StreamAwareQuickPool::~StreamAwareQuickPool()
{
  UMPIRE_LOG(Debug, "Releasing free blocks to device");
  m_is_destructing = true;
  release();
}

void StreamAwareQuickPool::allocate(void* stream, std::size_t bytes)
{
  // register stream ID in some global (let's start with vectors)
  unsigned int size = m_registered_streams.size();
  UMPIRE_LOG(Debug, "Size is: " << size);
  for (unsigned int i = 0; i < size; i++) {
    //if it does already contain stream ptr, then proceed with allocation
    if (m_registered_streams.at(i) == stream) {
      UMPIRE_LOG(Debug, "I found a registered stream, I am allocating bytes:  " << bytes );
      allocate(bytes);    
    } else {
      m_registered_streams.push_back(stream);
      cudaError_t status = cudaStreamQuery((cudaStream_t)stream);
      UMPIRE_LOG(Debug, "I did not find a registered stream so I am adding it to vector. My status here is " << status );
      if (status == cudaSuccess) {
        allocate(bytes);
        UMPIRE_LOG(Debug, "I added to vector and query status is complete, so I am allocating bytes:  " << bytes);
      }
      else if (status == cudaErrorNotReady) {
        //Maybe start with synchronize stream, then do something more efficient
        cudaStreamSynchronize((cudaStream_t)stream);
        UMPIRE_LOG(Debug, "I added to vector and my query status is not ready, so i synch'ed");
        //check to see if dealloc event has happened
      } else {
        std::cout << "Error!" << std::endl;
        UMPIRE_LOG(Debug, "Checking stream status didn't work"); 
        /*something went wrong*/
      }
    }
  }
  //if m_registered streams does not contain ID, add ID and use cudaStreamQuery.. maybe a cudaStreamSynch?

  //do something for streams
  //if different streams, same pool, and trying to put mem in same block, then...
  //check to see if dealloc has happened
}

void* StreamAwareQuickPool::allocate(std::size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");
  const std::size_t rounded_bytes{aligned_round_up(bytes)};
  const auto& best = m_size_map.lower_bound(rounded_bytes);

  Chunk* chunk{nullptr};

  if (best == m_size_map.end()) {
    std::size_t bytes_to_use{(m_actual_bytes == 0) ? m_first_minimum_pool_allocation_size
                                                   : m_next_minimum_pool_allocation_size};

    std::size_t size{(rounded_bytes > bytes_to_use) ? rounded_bytes : bytes_to_use};

    UMPIRE_LOG(Debug, "Allocating new chunk of size " << size);

    void* ret{nullptr};
    try {
#if defined(UMPIRE_ENABLE_BACKTRACE)
      {
        umpire::util::backtrace bt;
        umpire::util::backtracer<>::get_backtrace(bt);
        UMPIRE_LOG(Info, "actual_size:" << (m_actual_bytes + rounded_bytes) << " (prev: " << m_actual_bytes << ") "
                                        << umpire::util::backtracer<>::print(bt));
      }
#endif

      ret = aligned_allocate(size); // Will Poison
    } catch (...) {
      UMPIRE_LOG(Error,
                 "Caught error allocating new chunk, giving up free chunks and "
                 "retrying...");
      release();
      try {
        ret = aligned_allocate(size); // Will Poison
        UMPIRE_LOG(Debug, "memory reclaimed, chunk successfully allocated.");
      } catch (...) {
        UMPIRE_LOG(Error, "recovery failed.");
        throw;
      }
    }

    m_actual_bytes += size;
    m_releasable_bytes += size;
    m_releasable_blocks++;
    m_total_blocks++;
    m_actual_highwatermark = (m_actual_bytes > m_actual_highwatermark) ? m_actual_bytes : m_actual_highwatermark;

    void* chunk_storage{m_chunk_pool.allocate()};
    chunk = new (chunk_storage) Chunk{ret, size, size};
  } else {
    chunk = (*best).second;
    m_size_map.erase(best);
  }

  UMPIRE_LOG(Debug, "Using chunk " << chunk << " with data " << chunk->data << " and size " << chunk->size
                                   << " for allocation of size " << rounded_bytes);

  if ((chunk->size == chunk->chunk_size) && chunk->free) {
    m_releasable_bytes -= chunk->chunk_size;
    m_releasable_blocks--;
  }

  void* ret = chunk->data;
  m_pointer_map.insert(std::make_pair(ret, chunk));

  chunk->free = false;

  if (rounded_bytes != chunk->size) {
    std::size_t remaining{chunk->size - rounded_bytes};
    UMPIRE_LOG(Debug, "Splitting chunk " << chunk->size << "into " << rounded_bytes << " and " << remaining);

    void* chunk_storage{m_chunk_pool.allocate()};
    Chunk* split_chunk{new (chunk_storage)
                           Chunk{static_cast<char*>(ret) + rounded_bytes, remaining, chunk->chunk_size}};

    auto old_next = chunk->next;
    chunk->next = split_chunk;
    split_chunk->prev = chunk;
    split_chunk->next = old_next;

    if (split_chunk->next)
      split_chunk->next->prev = split_chunk;

    chunk->size = rounded_bytes;
    split_chunk->size_map_it = m_size_map.insert(std::make_pair(remaining, split_chunk));
  }

  m_current_bytes += rounded_bytes;

  UMPIRE_UNPOISON_MEMORY_REGION(m_allocator, ret, bytes);
  return ret;
}

void StreamAwareQuickPool::deallocate(void* stream, void* ptr, std::size_t size)
{
  //cast to cudastream_t
  //Do something with the stream
  deallocate(ptr, size);
  camp::resources::Event deallocate_has_occurred;
}

void StreamAwareQuickPool::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  auto chunk = (*m_pointer_map.find(ptr)).second;
  chunk->free = true;

  m_current_bytes -= chunk->size;

  UMPIRE_LOG(Debug, "Deallocating data held by " << chunk);

  UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, chunk->size);

  if (chunk->prev && chunk->prev->free == true) {
    auto prev = chunk->prev;
    UMPIRE_LOG(Debug, "Removing chunk" << prev << " from size map");

    m_size_map.erase(prev->size_map_it);

    prev->size += chunk->size;
    prev->next = chunk->next;

    if (prev->next)
      prev->next->prev = prev;

    UMPIRE_LOG(Debug, "Merging with prev" << prev << " and " << chunk);
    UMPIRE_LOG(Debug, "New size: " << prev->size);

    m_chunk_pool.deallocate(chunk);
    chunk = prev;
  }

  if (chunk->next && chunk->next->free == true) {
    auto next = chunk->next;
    chunk->size += next->size;
    chunk->next = next->next;
    if (chunk->next)
      chunk->next->prev = chunk;

    UMPIRE_LOG(Debug, "Merging with next" << chunk << " and " << next);
    UMPIRE_LOG(Debug, "New size: " << chunk->size);

    UMPIRE_LOG(Debug, "Removing chunk" << next << " from size map");
    m_size_map.erase(next->size_map_it);

    m_chunk_pool.deallocate(next);
  }

  UMPIRE_LOG(Debug, "Inserting chunk " << chunk << " with size " << chunk->size);

  if (chunk->size == chunk->chunk_size) {
    m_releasable_blocks++;
    m_releasable_bytes += chunk->chunk_size;
  }

  chunk->size_map_it = m_size_map.insert(std::make_pair(chunk->size, chunk));
  // can do this with iterator?
  m_pointer_map.erase(ptr);

  std::size_t suggested_size{m_should_coalesce(*this)};
  if (0 != suggested_size) {
    UMPIRE_LOG(Debug, "coalesce heuristic true, performing coalesce.");
    do_coalesce(suggested_size);
  }
}

//Check the event before releasing the pool? Should this have a check?
void StreamAwareQuickPool::release()
{
  //camp::resources::Resource resource = camp::resources::Host();
  UMPIRE_LOG(Debug, "() " << m_size_map.size() << " chunks in free map, m_is_destructing set to " << m_is_destructing);

#if defined(UMPIRE_ENABLE_BACKTRACE)
  std::size_t prev_size{m_actual_bytes};
#endif

  for (auto pair = m_size_map.begin(); pair != m_size_map.end();) {
    auto chunk = (*pair).second;
    UMPIRE_LOG(Debug, "Found chunk @ " << chunk->data);
    if ((chunk->size == chunk->chunk_size) && chunk->free) {
      UMPIRE_LOG(Debug, "Releasing chunk " << chunk->data);

      m_actual_bytes -= chunk->chunk_size;
      m_releasable_bytes -= chunk->chunk_size;
      m_releasable_blocks--;
      m_total_blocks--;

      try {
        //if(resource.get<camp::resources::Host>().get_platform() == camp::resources::Platform::host)
        //if(camp::resources::Host().get_event() != h_dealloc) ----> Need to make sure that no allocations are currently outstanding/pending first
           aligned_deallocate(chunk->data);
        //else
        //   UMPIRE_LOG(Debug, "camp resource types do not match");
      } catch (...) {
        if (m_is_destructing) {
          //
          // Ignore error in case the underlying vendor API has already shutdown
          //
          UMPIRE_LOG(Error, "Pool is destructing, runtime_error Ignored");
        } else {
          throw;
        }
      }

      m_chunk_pool.deallocate(chunk);
      pair = m_size_map.erase(pair);
    } else {
      ++pair;
    }
  }

#if defined(UMPIRE_ENABLE_BACKTRACE)
  if (prev_size > m_actual_bytes) {
    umpire::util::backtrace bt;
    umpire::util::backtracer<>::get_backtrace(bt);
    UMPIRE_LOG(Info, "actual_size:" << m_actual_bytes << " (prev: " << prev_size << ") "
                                    << umpire::util::backtracer<>::print(bt));
  }
#endif
}

std::size_t StreamAwareQuickPool::getReleasableBlocks() const noexcept
{
  return m_releasable_blocks;
}

std::size_t StreamAwareQuickPool::getTotalBlocks() const noexcept
{
  return m_total_blocks;
}

std::size_t StreamAwareQuickPool::getActualSize() const noexcept
{
  return m_actual_bytes;
}

std::size_t StreamAwareQuickPool::getCurrentSize() const noexcept
{
  return m_current_bytes;
}

std::size_t StreamAwareQuickPool::getReleasableSize() const noexcept
{
  return m_releasable_bytes;
}

std::size_t StreamAwareQuickPool::getActualHighwaterMark() const noexcept
{
  return m_actual_highwatermark;
}

Platform StreamAwareQuickPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits StreamAwareQuickPool::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

bool StreamAwareQuickPool::tracksMemoryUse() const noexcept
{
  return false;
}

std::size_t StreamAwareQuickPool::getBlocksInPool() const noexcept
{
  return m_pointer_map.size() + m_size_map.size();
}

std::size_t StreamAwareQuickPool::getLargestAvailableBlock() noexcept
{
  if (!m_size_map.size()) {
    return 0;
  }
  return m_size_map.rbegin()->first;
}

void StreamAwareQuickPool::coalesce() noexcept
{
  UMPIRE_LOG(Debug, "()");

  umpire::event::record([&](auto& event) {
    event.name("coalesce").category(event::category::operation).tag("allocator_name", getName()).tag("replay", "true");
  });

  std::size_t suggested_size{m_should_coalesce(*this)};
  if (0 != suggested_size) {
    UMPIRE_LOG(Debug, "coalesce heuristic true, performing coalesce, suggested size is " << suggested_size);
    do_coalesce(suggested_size);
  }
}

void StreamAwareQuickPool::do_coalesce(std::size_t suggested_size) noexcept
{
  //Event is needed for allocation, but we just want to send the allocate the same event.
  //camp::resources::Event event = camp::resources::Host().get_event();
  camp::resources::Resource resource = camp::resources::Host();
  if (m_size_map.size() > 1) {
    UMPIRE_LOG(Debug, "()");
    release();
    std::size_t size_post{getActualSize()};

    if (size_post < suggested_size) {
      std::size_t alloc_size{suggested_size - size_post};

      UMPIRE_LOG(Debug, "coalescing " << alloc_size << " bytes.");
      //Because this is a controlled, internal function, I may just be able to do a normal allocate and deallocate
      auto ptr = allocate(alloc_size);
      deallocate(ptr, alloc_size);
    }
  }
}

PoolCoalesceHeuristic<StreamAwareQuickPool> StreamAwareQuickPool::blocks_releasable(std::size_t nblocks)
{
  return
      [=](const strategy::StreamAwareQuickPool& pool) { return pool.getReleasableBlocks() >= nblocks ? pool.getActualSize() : 0; };
}

PoolCoalesceHeuristic<StreamAwareQuickPool> StreamAwareQuickPool::blocks_releasable_hwm(std::size_t nblocks)
{
  return [=](const strategy::StreamAwareQuickPool& pool) {
    return pool.getReleasableBlocks() >= nblocks ? pool.getHighWatermark() : 0;
  };
}

PoolCoalesceHeuristic<StreamAwareQuickPool> StreamAwareQuickPool::percent_releasable(int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR(
        runtime_error,
        umpire::fmt::format("Invalid percentage: {}, percentage must be an integer between 0 and 100", percentage));
  }
  if (percentage == 0) {
    return [=](const StreamAwareQuickPool& UMPIRE_UNUSED_ARG(pool)) { return 0; };
  } else if (percentage == 100) {
    return [=](const strategy::StreamAwareQuickPool& pool) {
      return pool.getActualSize() == pool.getReleasableSize() ? pool.getActualSize() : 0;
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);
    return [=](const strategy::StreamAwareQuickPool& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
      return pool.getReleasableSize() >= threshold ? pool.getActualSize() : 0;
    };
  }
}

PoolCoalesceHeuristic<StreamAwareQuickPool> StreamAwareQuickPool::percent_releasable_hwm(int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR(
        runtime_error,
        umpire::fmt::format("Invalid percentage: {}, percentage must be an integer between 0 and 100", percentage));
  }
  if (percentage == 0) {
    return [=](const StreamAwareQuickPool& UMPIRE_UNUSED_ARG(pool)) { return 0; };
  } else if (percentage == 100) {
    return [=](const strategy::StreamAwareQuickPool& pool) {
      return pool.getActualSize() == pool.getReleasableSize() ? pool.getHighWatermark() : 0;
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);
    return [=](const strategy::StreamAwareQuickPool& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
      return pool.getReleasableSize() >= threshold ? pool.getHighWatermark() : 0;
    };
  }
}

std::ostream& operator<<(std::ostream& out, umpire::strategy::PoolCoalesceHeuristic<StreamAwareQuickPool>&)
{
  return out;
}

} // end of namespace strategy
} // end namespace umpire
