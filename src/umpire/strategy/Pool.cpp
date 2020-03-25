//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/Pool.hpp"

#include "umpire/util/FixedMallocPool.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {
namespace strategy {

namespace {
  inline std::size_t aligned_size(const std::size_t size) {
    const std::size_t boundary = 16;
  //  return std::size_t (size + (boundary-1)) & ~(boundary-1);
    return size + boundary - 1 - (size - 1) % boundary;
  }
}

Pool::Pool(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::size_t initial_alloc_size,
    const std::size_t min_alloc_size,
    const int) noexcept :
  AllocationStrategy{name, id},
  m_pointer_map{},
  m_size_map{},
  m_chunk_pool{sizeof(Chunk)},
  m_allocator{allocator.getAllocationStrategy()},
  m_initial_alloc_bytes{initial_alloc_size},
  m_min_alloc_bytes{min_alloc_size}
  //m_align_bytes{align_bytes}
{
  void* ptr{m_allocator->allocate(m_initial_alloc_bytes)};
  m_actual_bytes += m_initial_alloc_bytes;

  void* chunk_storage{m_chunk_pool.allocate()};
  Chunk* chunk{new (chunk_storage) Chunk(ptr, initial_alloc_size, m_initial_alloc_bytes)};
  chunk->size_map_it = m_size_map.insert(std::make_pair(m_initial_alloc_bytes, chunk));
}

Pool::~Pool()
{
}

void* 
Pool::allocate(std::size_t bytes)
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
      ret = m_allocator->allocate(size);
      m_actual_bytes += size;
    } catch (...) {
      UMPIRE_LOG(Debug, "Caught error allocating");
    }
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
  return ret;
}

void 
Pool::deallocate(void* ptr)
{
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

  chunk->size_map_it = m_size_map.insert(std::make_pair(chunk->size, chunk));
  m_pointer_map.erase(ptr);
}

void Pool::release()
{
  UMPIRE_LOG(Debug, "release");
  UMPIRE_LOG(Debug, m_size_map.size() << " chunks in free map");

  for (auto pair = m_size_map.begin(); pair != m_size_map.end(); )
  {
    auto chunk = (*pair).second;
    UMPIRE_LOG(Debug, "Found chunk @ " << chunk->data);
    if ( (chunk-> size == chunk->chunk_size) 
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
}

std::size_t 
Pool::getCurrentSize() const noexcept
{
  return m_curr_bytes;
}

std::size_t 
Pool::getActualSize() const noexcept
{
  return m_actual_bytes;
}

std::size_t 
Pool::getHighWatermark() const noexcept
{
  return m_highwatermark;
}

Platform 
Pool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

void
Pool::coalesce() noexcept
{
  size_t size_pre{getActualSize()};
  release();
  size_t size_post{getActualSize()};
  size_t alloc_size{size_pre-size_post};
  auto ptr = allocate(alloc_size);
  deallocate(ptr);
}


} // end of namespace strategy
} // end namespace umpire
