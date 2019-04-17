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

#include "umpire/strategy/FixedPool.hpp"

#include "umpire/util/Macros.hpp"

#include <cstring>
#include <cstdlib>
#include <algorithm>

// TODO: Support for Windows
#include <strings.h>

namespace umpire {
namespace strategy {

static constexpr size_t bits_per_int = sizeof(int) * 8;

FixedPool::Pool::Pool(AllocationStrategy* allocation_strategy,
                      const size_t object_bytes, const size_t objects_per_pool) :
  strategy(allocation_strategy),
  data(reinterpret_cast<char*>(strategy->allocate(object_bytes * objects_per_pool))),
  avail(reinterpret_cast<int*>(std::malloc(objects_per_pool/bits_per_int + 1))),
  num_avail(objects_per_pool)
{
  std::memset(avail, ~0, objects_per_pool/bits_per_int + 1);
}

FixedPool::Pool::~Pool()
{
  strategy->deallocate(data);
  std::free(avail);
  data = nullptr;
  avail = nullptr;
  strategy = nullptr;
  num_avail = 0;
}

FixedPool::FixedPool(const std::string& name, int id,
                     Allocator allocator, const size_t object_bytes,
                     const size_t objects_per_pool) :
  AllocationStrategy(name, id),
  m_strategy(allocator.getAllocationStrategy()),
  m_obj_bytes(object_bytes),
  m_obj_per_pool(objects_per_pool),
  m_current_bytes(0),
  m_highwatermark(0),
  m_pool()
{
  newPool();
}

void
FixedPool::newPool()
{
  m_pool.emplace_back(m_strategy, m_obj_bytes, m_obj_per_pool);
}

void*
FixedPool::allocInPool(Pool& p) noexcept
{
  if (!p.num_avail) return nullptr;

  const int avail_bytes = m_obj_per_pool/bits_per_int + 1;

  for (int int_index = 0; int_index < avail_bytes; ++int_index) {
    const int bit_index = ffs(p.avail[int_index]) - 1;
    if (bit_index >= 0) {
      p.avail[int_index] ^= 1 << bit_index;
      p.num_avail--;
      const int offset = int_index * bits_per_int + bit_index;
      return reinterpret_cast<void*>(p.data + m_obj_bytes * offset);
    }
  }

  UMPIRE_ASSERT("FixedPool: Logic error in allocate");

  return nullptr;
}

void*
FixedPool::allocate(size_t bytes)
{
  void* ptr = nullptr;

  for (auto& p : m_pool) {
    ptr = allocInPool(p);
    if (ptr) break;
  }

  if (!ptr) {
    newPool();
    ptr = allocate(bytes);
  }

  UMPIRE_ASSERT(ptr);

  m_current_bytes += m_obj_bytes;
  m_highwatermark = std::max(m_highwatermark, m_current_bytes);

  return ptr;
}

void
FixedPool::deallocate(void* ptr)
{
  for (auto& p : m_pool) {
    const int object_index = (reinterpret_cast<char*>(ptr) - p.data) / m_obj_bytes;
    if (static_cast<unsigned int>(object_index) < m_obj_per_pool) {
      const int byte_index = object_index * m_obj_bytes;
      const int int_index = byte_index / bits_per_int;
      const short bit_index = byte_index % bits_per_int;

      UMPIRE_ASSERT(! (p.avail[int_index] & (1 << bit_index)));
      p.avail[int_index] ^= 1 << bit_index;
      p.num_avail++;

      return;
    }
  }
}

long
FixedPool::getCurrentSize() const noexcept
{
  return m_current_bytes;
}

long
FixedPool::getActualSize() const noexcept
{
  const int avail_bytes = m_obj_per_pool/bits_per_int + 1;
  return m_pool.size() * (m_obj_per_pool * m_obj_bytes + avail_bytes)
    + m_pool.capacity() * sizeof(Pool)
    + sizeof(FixedPool);
}

long
FixedPool::getHighWatermark() const noexcept
{
  return m_highwatermark;
}

Platform
FixedPool::getPlatform() noexcept
{
  return m_strategy->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
