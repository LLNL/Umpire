//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/detail/fixed_malloc_pool.hpp"
#include "umpire/detail/log.hpp"
#include "umpire/detail/macros.hpp"

#include <map>

namespace umpire {
namespace strategy {

namespace {
  inline std::size_t aligned_size(const std::size_t size) {
    const std::size_t boundary = 16;
  //  return std::size_t (size + (boundary-1)) & ~(boundary-1);
    return size + boundary - 1 - (size - 1) % boundary;
  }
}

template<typename Memory=memory, bool Tracking=true>
class quick_pool :
  public allocation_strategy
{
  public:

  using platform = typename Memory::platform; 
  using Pointer = void*;
  using CoalesceHeuristic = std::function<bool (const strategy::quick_pool<Memory, Tracking>& )>;

  //static CoalesceHeuristic percent_releasable(int percentage);
static CoalesceHeuristic
percent_releasable(int percentage)
{
  if ( percentage < 0 || percentage > 100 ) {
    UMPIRE_ERROR("Invalid percentage of " << percentage 
        << ", percentage must be an integer between 0 and 100");
  }

  if ( percentage == 0 ) {
    return [=] (const quick_pool<Memory, Tracking>&) {
        return false;
    };
  } else if ( percentage == 100 ) {
    return [=] (const strategy::quick_pool<Memory, Tracking>& pool) {
        return (pool.get_current_size() == 0 && pool.getReleasableSize() > 0);
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);

    return [=] (const strategy::quick_pool<Memory, Tracking>& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.get_actual_size());
      return (pool.getReleasableSize() >= threshold);
    };
  }
}

quick_pool(
    const std::string& name,
    Memory* memory,
    const std::size_t initial_alloc_size = 512*1024*1024,
    const std::size_t min_alloc_size = 1024*1024,
    CoalesceHeuristic coalesce_heuristic = percent_releasable(100)) noexcept :
  allocation_strategy{name},
  m_pointer_map{},
  m_size_map{},
  m_chunk_pool{sizeof(Chunk)},
  m_should_coalesce{coalesce_heuristic},
  m_initial_alloc_bytes{initial_alloc_size},
  m_min_alloc_bytes{min_alloc_size},
  m_allocator{memory}
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
  m_actual_bytes += m_initial_alloc_bytes;
  m_releasable_bytes += m_initial_alloc_bytes;

  void* chunk_storage{m_chunk_pool.allocate()};
  Chunk* chunk{new (chunk_storage) Chunk(ptr, initial_alloc_size, m_initial_alloc_bytes)};
  chunk->size_map_it = m_size_map.insert(std::make_pair(m_initial_alloc_bytes, chunk));
}

~quick_pool()
{
}

void* 
allocate(std::size_t bytes) final
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

  m_curr_bytes += bytes;
  if constexpr(Tracking) {
    return this->track_allocation(this, ret, bytes);
  } else {
    return ret;
  }
}

void 
deallocate(void* ptr)
{
  if constexpr(Tracking) {
    this->untrack_allocation(ptr);
  }

  UMPIRE_LOG(Debug, "deallocate(" << ptr << ")");
  auto chunk = (*m_pointer_map.find(ptr)).second;
  chunk->free = true;
  m_curr_bytes -= chunk->size;

  UMPIRE_LOG(Debug, "Deallocating data held by " << chunk);

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

void release()
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
get_actual_size() const noexcept
{
  return m_actual_bytes;
}

std::size_t
getReleasableSize() const noexcept
{
  if (m_size_map.size() > 1)
    return m_releasable_bytes;
  else return 0;
}

camp::resources::Platform 
get_platform() noexcept
{
  return m_allocator->get_platform();
}

void
coalesce() noexcept
{
  std::size_t size_pre{get_actual_size()};
  release();
  std::size_t size_post{get_actual_size()};
  std::size_t alloc_size{size_pre-size_post};
  UMPIRE_LOG(Debug, "coalescing " << alloc_size << " bytes.");
  auto ptr = allocate(alloc_size);
  deallocate(ptr);
}


private:
  private:
    struct Chunk;

    template <typename Value>
    class pool_allocator {
      public:
        using value_type = Value;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        pool_allocator() :
          pool{new detail::fixed_malloc_pool{sizeof(Value)}} {}

        /// BUG: Only required for MSVC
        template<typename U>
        pool_allocator(const pool_allocator<U>& other): 
          pool{other.pool}
        {}

        Value* allocate(std::size_t n) {
          return static_cast<Value*>(pool->allocate(n));
        }

        void deallocate(Value* data, std::size_t)
        {
          pool->deallocate(data);
        }

      detail::fixed_malloc_pool* pool;
    };

    using pointer_map = std::unordered_map<void*, Chunk*>;
    using size_map = std::multimap<std::size_t, Chunk*, std::less<std::size_t>, pool_allocator<std::pair<const std::size_t, Chunk*>>>;

    struct Chunk {
      Chunk(void* ptr, std::size_t s, std::size_t cs) :
        data{ptr}, size{s}, chunk_size{cs} {}

      void* data{nullptr};
      std::size_t size{0};
      std::size_t chunk_size{0};
      bool free{true};
      Chunk* prev{nullptr};
      Chunk* next{nullptr};
      typename size_map::iterator size_map_it;
    };

    pointer_map m_pointer_map;
    size_map m_size_map;

    detail::fixed_malloc_pool m_chunk_pool;
    CoalesceHeuristic m_should_coalesce;

    const std::size_t m_initial_alloc_bytes;
    const std::size_t m_min_alloc_bytes;

    std::size_t m_curr_bytes{0};
    std::size_t m_actual_bytes{0};
    std::size_t m_highwatermark{0};
    std::size_t m_releasable_bytes{0};

    Memory* m_allocator;
};


} // end of namespace strategy
} // end namespace umpire
