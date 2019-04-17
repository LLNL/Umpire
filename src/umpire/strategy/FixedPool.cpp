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

#include "umpire/util/Macros.hpp"

#include <memory>
#include <cstring>

// TODO: Support for Windows
#include <strings.h>

namespace umpire {
namespace strategy {

FixedPool(const std::string& name, int id,
          Allocator allocator, const size_t object_size,
          const size_t objects_per_pool) :
  AllocationStrategy(name, id),
  m_strategy(allocator.getAllocationStrategy()),
  m_avail_bytes(num_objects/bits_per_int + 1),
  m_obj_bytes(object_size),
  m_num_obj(num_objects)
{
}

FixedPool::Pool::Pool(const size_t object_size, const size_t num_objects,
                      FixedPool* fp) :
  data(reinterpret_cast<char*>(fp->m_strategy->allocate(fp->m_obj_bytes * fp->m_num_obj))),
  avail(reinterpret_cast<int*>(std::malloc(fp->m_avail_bytes))),
  num_avail(fp->m_num_objs)
{
  std::memset(avail, ~0, fp->m_avail_bytes);
}

void*
FixedPool::allocate(size_t bytes)
{
  T* ptr = nullptr;

  for (size_t i = m_pool.size(); i-- > 0; ) allocInPool(m_pool[i]);

  if (!ptr) {
    m_pool.emplace_back(m_obj_bytes, m_num_obj, this);
    ptr = allocate(bytes);
  }

  UMPIRE_ASSERT(ptr);

  return ptr;
}

void*
FixedPool::allocInPool(Pool& p)
{
  if (!p.num_avail) return nullptr;

  for (int a = 0; a < m_abytes; ++a) {
    const int bit = ffs(p.avail[i]) - 1;
    if (bit >= 0) {
      p.avail[i] ^= 1 << bit;
      p.num_avail--;
      const int index = a * bits_per_int + bit;
      return reinterpret_cast<void*>(p.data + m_obj_size * index);
    }
  }

  UMPIRE_ERROR("FixedPool: Logic error in allocate");
}

void
FixedPool::deallocate(void* ptr)
{

  for (const auto& p : m_pool) {
    const int object_index = (reinterpret_cast<char*>(ptr) - p.data) / m_obj_bytes;
    if (static_cast<unsigned int>(object_index) < m_num_objs) {
      const int byte_index = object_index * m_obj_bytes;
      const int int_index = byte_index / bits_per_int;
      const short bit_index = byte_index % bits_per_int;

      UMPIRE_ASSERT(p.avail[int_index] & (1 << bit_index));
      p.avail[int_index] ^= 1 << byte_index;
      p.num_avail++;

      return;
    }
  }
}

long
FixedPool::getCurrentSize() const noexcept
{
  return m_current_size;
}

long
FixedPool::getActualSize() const noexcept
{
  return m_total_pool_size;
}

long
FixedPool::getHighWatermark() const noexcept
{
  return m_highwatermark;
}

long
FixedPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

} // end of namespace strategy
} // end of namespace umpire
