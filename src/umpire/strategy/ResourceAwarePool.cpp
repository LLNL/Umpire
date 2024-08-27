// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/ResourceAwarePool.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/strategy/PoolCoalesceHeuristic.hpp"
#include "umpire/strategy/mixins/AlignedAllocation.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/event/event.hpp"
#include "umpire/util/memory_sanitizers.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
using resource_type = camp::resources::Cuda;
#elif defined(UMPIRE_ENABLE_HIP)
using resource_type = camp::resources::Hip;
#endif

namespace umpire {
namespace strategy {

ResourceAwarePool::ResourceAwarePool(const std::string& name, int id, Allocator allocator,
                                     const std::size_t first_minimum_pool_allocation_size,
                                     const std::size_t next_minimum_pool_allocation_size, std::size_t alignment,
                                     PoolCoalesceHeuristic<ResourceAwarePool> should_coalesce) noexcept
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "ResourceAwarePool"},
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

ResourceAwarePool::~ResourceAwarePool()
{
  UMPIRE_LOG(Debug, "Releasing free blocks to device");
  m_is_destructing = true;
  release();
}

void* ResourceAwarePool::allocate(std::size_t UMPIRE_UNUSED_ARG(bytes))
{
  void* ptr{nullptr};
  UMPIRE_ERROR(runtime_error, fmt::format("Don't call this function!"));
  return ptr;
}

void* ResourceAwarePool::allocate_resource(camp::resources::Resource r, std::size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");
  const std::size_t rounded_bytes{aligned_round_up(bytes)};
  const auto& best = m_free_map.lower_bound(rounded_bytes);

  Chunk* chunk{nullptr};

  // auto pending_chunks_exist = m_pending_map.find(r);

  if (!m_pending_map.empty()) {
    for (auto pending_chunk : m_pending_map) {
      if (pending_chunk->m_event.check()) // no longer pending
      {
        do_deallocate(pending_chunk); // TODO: can I erase it from the list then?
      }
      if (pending_chunk->size >= rounded_bytes && pending_chunk->m_resource == r) {
        chunk->size = pending_chunk->size; // is this necessary?
        chunk = pending_chunk;
        chunk->free = false;
        // TODO: Do I add to actual bytes, releasable blocks, total blocks, etc.????
        std::size_t bytes_to_use{(m_actual_bytes == 0) ? m_first_minimum_pool_allocation_size
                                                       : m_next_minimum_pool_allocation_size};
        std::size_t size{(rounded_bytes > bytes_to_use) ? rounded_bytes : bytes_to_use};
        m_actual_bytes += size;
        m_releasable_bytes += size;
        m_releasable_blocks++;
        m_total_blocks++;
        m_actual_highwatermark = (m_actual_bytes > m_actual_highwatermark) ? m_actual_bytes : m_actual_highwatermark;
        break;
      }
    }
  }

  if (chunk == nullptr) {
    if (best == m_free_map.end()) {
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
      chunk = new (chunk_storage) Chunk{ret, size, size, r};
    } else {
      chunk = (*best).second;
      m_free_map.erase(best);
    }
  }

  UMPIRE_LOG(Debug, "Using chunk " << chunk << " with data " << chunk->data << " and size " << chunk->size
                                   << " for allocation of size " << rounded_bytes);

  if ((chunk->size == chunk->chunk_size) && chunk->free) {
    m_releasable_bytes -= chunk->chunk_size;
    m_releasable_blocks--;
  }

  void* ret = chunk->data;
  m_used_map.insert(std::make_pair(ret, chunk));

  chunk->free = false;

  if (rounded_bytes != chunk->size) {
    std::size_t remaining{chunk->size - rounded_bytes};
    UMPIRE_LOG(Debug, "Splitting chunk " << chunk->size << "into " << rounded_bytes << " and " << remaining);

    void* chunk_storage{m_chunk_pool.allocate()};
    Chunk* split_chunk{new (chunk_storage)
                           Chunk{static_cast<char*>(ret) + rounded_bytes, remaining, chunk->chunk_size, r}};
    auto old_next = chunk->next;
    chunk->next = split_chunk;
    split_chunk->prev = chunk;
    split_chunk->next = old_next;

    if (split_chunk->next)
      split_chunk->next->prev = split_chunk;

    chunk->size = rounded_bytes;
    split_chunk->size_map_it = m_free_map.insert(std::make_pair(remaining, split_chunk));
  }

  m_current_bytes += rounded_bytes;

  UMPIRE_UNPOISON_MEMORY_REGION(m_allocator, ret, bytes);
  return ret;
}

void ResourceAwarePool::deallocate(void* ptr, std::size_t size)
{
  auto r = getResource(ptr);

  UMPIRE_LOG(Warning, fmt::format("You called deallocate with no resource. Calling deallocate with the resource ",
                                    "returned by getResource: {}.", camp::resources::to_string(r)));
  deallocate_resource(r, ptr, size);
}

void ResourceAwarePool::do_deallocate(Chunk* chunk) noexcept
{
  UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, chunk->size);

  UMPIRE_LOG(Debug, "In the do_deallocate function. Deallocating data held by " << chunk);

  if (chunk->prev && chunk->prev->free == true) {
    auto prev = chunk->prev;
    UMPIRE_LOG(Debug, "Removing chunk" << prev << " from size map");

    m_free_map.erase(prev->size_map_it);

    prev->size += chunk->size;
    prev->next = chunk->next;

    prev->m_event = chunk->m_event;
    prev->m_resource = chunk->m_resource;

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
    m_free_map.erase(next->size_map_it);

    m_chunk_pool.deallocate(next);
  }

  UMPIRE_LOG(Debug, "Inserting chunk " << chunk << " with size " << chunk->size);

  if (chunk->size == chunk->chunk_size) {
    m_releasable_blocks++;
    m_releasable_bytes += chunk->chunk_size;
  }

  chunk->size_map_it = m_free_map.insert(std::make_pair(chunk->size, chunk));
  chunk->free = true;
}

void ResourceAwarePool::deallocate_resource(camp::resources::Resource r, void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  // UMPIRE_LOG(Debug, "(Resource=" << r << ")");
  auto chunk = (*m_used_map.find(ptr)).second;

  // TODO: if( !chunk) --> isn't this a error to check for?

  auto my_r = getResource(ptr);
  if (my_r != r) {
    UMPIRE_LOG(Warning, fmt::format("Called deallocate with different resource than what is returned by getResource. Called with {},", 
                                      "but getResource returned: {}", camp::resources::to_string(r), camp::resources::to_string(my_r)));
    UMPIRE_LOG(Debug, fmt::format("getResource doesn't match resource passed to deallocate. Resource used: {} .", camp::resources::to_string(r)));
  }

  // chunk is now pending
  m_pending_map.push_back(chunk);
  chunk->m_event = r.get_event();

  m_used_map.erase(ptr);
  m_current_bytes -= chunk->size;

  // Call deallocate logic only for a non-pending chunk
  if (chunk->m_event.check()) {
    do_deallocate(chunk);
  }

  std::size_t suggested_size{m_should_coalesce(*this)};
  if (0 != suggested_size) {
    UMPIRE_LOG(Debug, "coalesce heuristic true, performing coalesce.");
    do_coalesce(suggested_size);
  }
}

void ResourceAwarePool::release()
{
  UMPIRE_LOG(Debug, "() " << m_free_map.size() << " chunks in free map, m_is_destructing set to " << m_is_destructing);

#if defined(UMPIRE_ENABLE_BACKTRACE)
  std::size_t prev_size{m_actual_bytes};
#endif

  // This will check all chunks in m_pending_map and erase the entry if event is complete
  for (auto chunk = m_pending_map.begin(); chunk != m_pending_map.end(); chunk++) {
    if ((*chunk) != nullptr && (*chunk)->m_event.check()) {
      m_pending_map.erase(chunk);
      m_free_map.insert(std::make_pair((*chunk)->size, (*chunk))); // Make sure this is correct!
      (*chunk)->free = true;                                       // Is free up to date everywhere else too?
    }
  }

  for (auto pair = m_free_map.begin(); pair != m_free_map.end();) {
    auto chunk = (*pair).second;
    UMPIRE_LOG(Debug, "Found chunk @ " << chunk->data);
    if ((chunk->size == chunk->chunk_size) && chunk->free) {
      UMPIRE_LOG(Debug, "Releasing chunk " << chunk->data);

      m_actual_bytes -= chunk->chunk_size;
      m_releasable_bytes -= chunk->chunk_size;
      m_releasable_blocks--;
      m_total_blocks--;

      try {
        aligned_deallocate(chunk->data);
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
      pair = m_free_map.erase(pair);
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

std::size_t ResourceAwarePool::getReleasableBlocks() const noexcept
{
  return m_releasable_blocks;
}

std::size_t ResourceAwarePool::getTotalBlocks() const noexcept
{
  return m_total_blocks;
}

std::size_t ResourceAwarePool::getPendingSize() const noexcept
{
  return m_pending_map.size();
}

std::size_t ResourceAwarePool::getActualSize() const noexcept
{
  return m_actual_bytes;
}

std::size_t ResourceAwarePool::getCurrentSize() const noexcept
{
  return m_current_bytes;
}

std::size_t ResourceAwarePool::getReleasableSize() const noexcept
{
  return m_releasable_bytes;
}

std::size_t ResourceAwarePool::getActualHighwaterMark() const noexcept
{
  return m_actual_highwatermark;
}

Platform ResourceAwarePool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

camp::resources::Resource ResourceAwarePool::getResource(void* ptr) const
{
  auto it = m_used_map.find(ptr); // check used chunks
  if (it != m_used_map.end()) {
    auto chunk = it->second;
    return chunk->m_resource;
  }
  for (auto& chunk : m_pending_map) { // chunk pending chunks
    if (chunk->data == ptr) {
      return chunk->m_resource;
    }
  }
  // UMPIRE_ERROR(runtime_error, fmt::format("BANANAS!!"));
  // TODO: if it is free, do we care what resource it has?
  return camp::resources::Host{}; // If we get here, the chunk is free
}

MemoryResourceTraits ResourceAwarePool::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

bool ResourceAwarePool::tracksMemoryUse() const noexcept
{
  return false;
}

std::size_t ResourceAwarePool::getBlocksInPool() const noexcept
{
  return m_used_map.size() + m_free_map.size();
}

std::size_t ResourceAwarePool::getLargestAvailableBlock() noexcept
{
  if (!m_free_map.size()) {
    return 0;
  }
  return m_free_map.rbegin()->first;
}

void ResourceAwarePool::coalesce() noexcept
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

void ResourceAwarePool::do_coalesce(std::size_t suggested_size) noexcept
{
  if (m_free_map.size() > 1) {
    UMPIRE_LOG(Debug, "()");
    release();
    std::size_t size_post{getActualSize()};

    if (size_post < suggested_size) {
      std::size_t alloc_size{suggested_size - size_post};

      camp::resources::Resource r = camp::resources::Host().get_default();
      UMPIRE_LOG(Debug, "coalescing " << alloc_size << " bytes.");
      auto ptr = allocate_resource(r, alloc_size);
      deallocate_resource(r, ptr, alloc_size);
    }
  }
}

PoolCoalesceHeuristic<ResourceAwarePool> ResourceAwarePool::blocks_releasable(std::size_t nblocks)
{
  return [=](const strategy::ResourceAwarePool& pool) {
    return pool.getReleasableBlocks() >= nblocks ? pool.getActualSize() : 0;
  };
}

PoolCoalesceHeuristic<ResourceAwarePool> ResourceAwarePool::blocks_releasable_hwm(std::size_t nblocks)
{
  return [=](const strategy::ResourceAwarePool& pool) {
    return pool.getReleasableBlocks() >= nblocks ? pool.getHighWatermark() : 0;
  };
}

PoolCoalesceHeuristic<ResourceAwarePool> ResourceAwarePool::percent_releasable(int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("Invalid percentage: {}, percentage must be an integer between 0 and 100", percentage));
  }
  if (percentage == 0) {
    return [=](const ResourceAwarePool& UMPIRE_UNUSED_ARG(pool)) { return 0; };
  } else if (percentage == 100) {
    return [=](const strategy::ResourceAwarePool& pool) {
      return pool.getActualSize() == pool.getReleasableSize() ? pool.getActualSize() : 0;
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);
    return [=](const strategy::ResourceAwarePool& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
      return pool.getReleasableSize() >= threshold ? pool.getActualSize() : 0;
    };
  }
}

PoolCoalesceHeuristic<ResourceAwarePool> ResourceAwarePool::percent_releasable_hwm(int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("Invalid percentage: {}, percentage must be an integer between 0 and 100", percentage));
  }
  if (percentage == 0) {
    return [=](const ResourceAwarePool& UMPIRE_UNUSED_ARG(pool)) { return 0; };
  } else if (percentage == 100) {
    return [=](const strategy::ResourceAwarePool& pool) {
      return pool.getActualSize() == pool.getReleasableSize() ? pool.getHighWatermark() : 0;
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);
    return [=](const strategy::ResourceAwarePool& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
      return pool.getReleasableSize() >= threshold ? pool.getHighWatermark() : 0;
    };
  }
}

std::ostream& operator<<(std::ostream& out, umpire::strategy::PoolCoalesceHeuristic<ResourceAwarePool>&)
{
  return out;
}

} // end of namespace strategy
} // end namespace umpire
