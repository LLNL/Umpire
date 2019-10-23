//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/Pool.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {
namespace strategy {

Pool::Pool(
    const std::string& name,
    int id,
    Allocator allocator,
    const std::size_t initial_alloc_size,
    const std::size_t min_alloc_size,
    const int align_bytes) noexcept :
  AllocationStrategy{name, id},
  m_pointer_map{},
  m_size_map{},
  m_allocator{allocator.getAllocationStrategy()},
  m_initial_alloc_bytes{initial_alloc_size},
  m_min_alloc_bytes{min_alloc_size},
  m_align_bytes{align_bytes}
{
  void* ptr{m_allocator->allocate(initial_alloc_size)};
  auto chunk{new Chunk(ptr, initial_alloc_size, initial_alloc_size, true, nullptr, nullptr)};

  m_size_map[initial_alloc_size] = chunk;
}

Pool::~Pool()
{
}

void* 
Pool::allocate(std::size_t bytes) override
{
  const auto& best = m_size_map.lower_bound(bytes);

  if (best == m_size_map.end()) {

    std::size_t size{round_up(bytes)};
    void* ret{m_allocator->allocate(size)};
    auto chunk{new Chunk{ptr, bytes, size, false}};
    m_ptr_map[ret] = chunk;
    return ptr;

  } else {
    auto chunk = *best;
    void* ret = chunk->data;
    m_pointer_map[ret] = chunk;
    m_size_map.erase(best);
    chunk->free = false;

    if (bytes != chunk.size) {
      std::size_t remaining{chunk->size - bytes};
      auto split_chunk{new Chunk{ptr+bytes, remaining, chunk->size, true, chunk}};
      chunk->next = split_chunk;
      m_size_map[remaining] = split_chunk;
    }

    return ret;
  }
}

void 
Pool::deallocate(void* ptr) override
{
  auto chunk = m_pointer_map[ptr];
  chunk->free = true;

  if (chunk->next) {
  }

  if (chunk->prev) {
  }
}

void Pool::release() override
{
}

std::size_t 
Pool::getCurrentSize() const noexcept override
{
}

std::size_t 
Pool::getActualSize() const noexcept override
{
}

std::size_t 
Pool::getHighWatermark() const noexcept override
{
}

Platform 
Pool::getPlatform() noexcept override
{
}


} // end of namespace strategy
} // end namespace umpire
