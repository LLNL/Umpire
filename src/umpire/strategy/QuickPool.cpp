//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/QuickPool.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/util/FixedMallocPool.hpp"
#include "umpire/util/memory_sanitizers.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

namespace {
  inline std::size_t aligned_size(const std::size_t size) {
    const std::size_t boundary = 16;
  //  return std::size_t (size + (boundary-1)) & ~(boundary-1);
    return size + boundary - 1 - (size - 1) % boundary;
  }
}

QuickPool::QuickPool(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::size_t initial_alloc_size,
    const std::size_t min_alloc_size,
    CoalesceHeuristic coalesce_heuristic) noexcept :
  AllocationStrategy{name, id},
  m_pointer_map{},
  m_size_map{},
  m_chunk_pool{sizeof(Chunk)},
  m_allocator{allocator.getAllocationStrategy()},
  m_should_coalesce{coalesce_heuristic},
  m_initial_alloc_bytes{initial_alloc_size},
  m_min_alloc_bytes{min_alloc_size}
{
#if defined(UMPIRE_ENABLE_BACKTRACE)
  {
    umpire::util::backtrace bt{};
    umpire::util::backtracer<>::get_backtrace(bt);
    UMPIRE_LOG(Info, "actual_size:"
      << m_initial_alloc_bytes << " (prev: 0) "
      << umpire::util::backtracer<>::print(bt));
  }
#endif

  void* ptr{m_allocator->allocate(m_initial_alloc_bytes)};
  UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, m_initial_alloc_bytes);
  m_actual_bytes += m_initial_alloc_bytes;
  m_releasable_bytes += m_initial_alloc_bytes;

  void* chunk_storage{m_chunk_pool.allocate()};
  Chunk* chunk{new (chunk_storage) Chunk(ptr, initial_alloc_size, m_initial_alloc_bytes)};
  chunk->size_map_it = m_size_map.insert(std::make_pair(m_initial_alloc_bytes, chunk));
}

QuickPool::~QuickPool()
{
  release(); 
}

void*
QuickPool::allocate(std::size_t bytes)
{
  UMPIRE_LOG(Debug, "allocate(" << bytes << ")");
  bytes = aligned_size(bytes);

  const auto& best = m_size_map.lower_bound(bytes);

  Chunk* chunk{nullptr};

  if (best == m_size_map.end()) {
    std::size_t size{ (bytes > m_min_alloc_bytes) ? bytes : m_min_alloc_bytes};
    UMPIRE_LOG(Debug, "Allocating new chunk of size " << size);

    void* ret{nullptr};
    try {
#if defined(UMPIRE_ENABLE_BACKTRACE)
      {
        umpire::util::backtrace bt{};
        umpire::util::backtracer<>::get_backtrace(bt);
        UMPIRE_LOG(Info, "actual_size:" << (m_actual_bytes+bytes)
          << " (prev: " << m_actual_bytes
          << ") " << umpire::util::backtracer<>::print(bt));
      }
#endif
      ret = m_allocator->allocate(size);
    } catch (...) {
      UMPIRE_LOG(Error, "Caught error allocating new chunk, giving up free chunks and retrying...");
      release();
      try {
        ret = m_allocator->allocate(size);
        UMPIRE_LOG(Debug, "memory reclaimed, chunk successfully allocated.");
      } catch (...) {
        UMPIRE_LOG(Error, "recovery failed.");
        throw;
      }
    }

    UMPIRE_POISON_MEMORY_REGION(m_allocator, ret, size);
    m_actual_bytes += size;
    m_releasable_bytes += size;
    void* chunk_storage{m_chunk_pool.allocate()};
    chunk = new (chunk_storage) Chunk{ret, size, size};
  } else {
    chunk = (*best).second;
    m_size_map.erase(best);
  }

  UMPIRE_LOG(Debug, "Using chunk " << chunk << " with data "
      << chunk->data << " and size " << chunk->size
      << " for allocation of size " << bytes);

  void* ret = chunk->data;
  m_pointer_map.insert(std::make_pair(ret, chunk));

  chunk->free = false;

  if (bytes != chunk->size) {
    std::size_t remaining{chunk->size - bytes};
    UMPIRE_LOG(Debug, "Splitting chunk " << chunk->size << "into "
        << bytes << " and " << remaining);

    m_releasable_bytes -= chunk->chunk_size;

    void* chunk_storage{m_chunk_pool.allocate()};
    Chunk* split_chunk{
      new (chunk_storage)
        Chunk{static_cast<char*>(ret)+bytes, remaining, chunk->chunk_size}};

    auto old_next = chunk->next;
    chunk->next = split_chunk;
    split_chunk->prev = chunk;
    split_chunk->next = old_next;

    if (split_chunk->next)
      split_chunk->next->prev = split_chunk;

    chunk->size = bytes;
    split_chunk->size_map_it =
      m_size_map.insert(std::make_pair(remaining, split_chunk));
  }

  UMPIRE_UNPOISON_MEMORY_REGION(m_allocator, ret, bytes);
  return ret;
}

void
QuickPool::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "deallocate(" << ptr << ")");
  auto chunk = (*m_pointer_map.find(ptr)).second;
  chunk->free = true;

  UMPIRE_LOG(Debug, "Deallocating data held by " << chunk);

  UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, chunk->size);

  if (chunk->prev && chunk->prev->free == true)
  {
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

  if ( chunk->next && chunk->next->free == true)
  {
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

  UMPIRE_LOG(Debug, "Inserting chunk " << chunk
      << " with size " << chunk->size);

  if (chunk->size == chunk->chunk_size) {
    m_releasable_bytes += chunk->chunk_size;
  }

  chunk->size_map_it = m_size_map.insert(std::make_pair(chunk->size, chunk));
  // can do this with iterator?
  m_pointer_map.erase(ptr);

  if (m_should_coalesce(*this))   {
    UMPIRE_LOG(Debug, "coalesce heuristic true, performing coalesce.");
    coalesce();
  }
}

void QuickPool::release()
{
  UMPIRE_LOG(Debug, "release");
  UMPIRE_LOG(Debug, m_size_map.size() << " chunks in free map");

  std::size_t prev_size{m_actual_bytes};

  for (auto pair = m_size_map.begin(); pair != m_size_map.end(); )
  {
    auto chunk = (*pair).second;
    UMPIRE_LOG(Debug, "Found chunk @ " << chunk->data);
    if ( (chunk->size == chunk->chunk_size)
        && chunk->free) {
      UMPIRE_LOG(Debug, "Releasing chunk " << chunk->data);
      UMPIRE_POISON_MEMORY_REGION(m_allocator, chunk->data, chunk->size);
      m_actual_bytes -= chunk->chunk_size;
      m_allocator->deallocate(chunk->data);
      m_chunk_pool.deallocate(chunk);
      pair = m_size_map.erase(pair);
    } else {
      ++pair;
    }
  }

#if defined(UMPIRE_ENABLE_BACKTRACE)
  if (prev_size > m_actual_bytes) {
    umpire::util::backtrace bt{};
    umpire::util::backtracer<>::get_backtrace(bt);
    UMPIRE_LOG(Info, "actual_size:" << m_actual_bytes
      << " (prev: " << prev_size
      << ") " << umpire::util::backtracer<>::print(bt));
  }
#else
  UMPIRE_USE_VAR(prev_size);
#endif
}

std::size_t
QuickPool::getActualSize() const noexcept
{
  return m_actual_bytes;
}

std::size_t
QuickPool::getReleasableSize() const noexcept
{
  if (m_size_map.size() > 1)
    return m_releasable_bytes;
  else return 0;
}

Platform
QuickPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void
QuickPool::coalesce() noexcept
{
  std::size_t size_pre{getActualSize()};
  release();
  std::size_t size_post{getActualSize()};
  std::size_t alloc_size{size_pre-size_post};

  //
  // Only perform the coalesce if there were bytes found to coalesce
  //
  if (alloc_size) {
    UMPIRE_LOG(Debug, "coalescing " << alloc_size << " bytes.");
    auto ptr = allocate(alloc_size);
    deallocate(ptr);
  }
}

QuickPool::CoalesceHeuristic
QuickPool::percent_releasable(int percentage)
{
  if ( percentage < 0 || percentage > 100 ) {
    UMPIRE_ERROR("Invalid percentage of " << percentage
        << ", percentage must be an integer between 0 and 100");
  }

  if ( percentage == 0 ) {
    return [=] (const QuickPool& UMPIRE_UNUSED_ARG(pool)) {
        return false;
    };
  } else if ( percentage == 100 ) {
    return [=] (const strategy::QuickPool& pool) {
        return ( pool.getActualSize() == pool.getReleasableSize() );
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);

    return [=] (const strategy::QuickPool& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
      return (pool.getReleasableSize() >= threshold);
    };
  }
}


} // end of namespace strategy
} // end namespace umpire
