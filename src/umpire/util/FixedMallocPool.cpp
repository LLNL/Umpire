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

#include "umpire/util/FixedMallocPool.hpp"

#include "umpire/util/Macros.hpp"

#include <cstring>
#include <cstdlib>
#include <algorithm>

#if !defined(_MSC_VER)
#define  _XOPEN_SOURCE_EXTENDED 1
#include <strings.h>
#endif

static int find_first_set(int i)
{
#if defined(_MSC_VER)
  unsigned long bit;
  unsigned long i_l = static_cast<unsigned long>(i);
  _BitScanForward(&bit, i_l);
  return static_cast<int>(bit);
#else
  return ffs(i);
#endif
}

namespace umpire {
namespace util {

static constexpr size_t bits_per_int = sizeof(int) * 8;

FixedMallocPool::Pool::Pool(const size_t object_bytes, const size_t objects_per_pool,
                            const size_t avail_bytes) :
  data(reinterpret_cast<char*>(std::malloc(object_bytes * objects_per_pool + avail_bytes))),
  avail(reinterpret_cast<int*>(data + object_bytes * objects_per_pool)),
  num_avail(objects_per_pool)
{
  // Set all bits to 1
  const unsigned char not_zero = ~0;
  std::memset(avail, not_zero, avail_bytes);
}

FixedMallocPool::FixedMallocPool(const size_t object_bytes,
                                 const size_t objects_per_pool) :
  m_obj_bytes(object_bytes),
  m_obj_per_pool(objects_per_pool),
  m_data_bytes(m_obj_bytes * m_obj_per_pool),
  m_avail_length(objects_per_pool/bits_per_int + 1),
  m_current_bytes(0),
  m_highwatermark(0),
  m_pool()
{
  newPool();
}

FixedMallocPool::~FixedMallocPool()
{
  for (auto& a : m_pool) std::free(a.data);
}

void
FixedMallocPool::newPool()
{
  m_pool.emplace_back(m_obj_bytes, m_obj_per_pool, m_avail_length * sizeof(int));
}

void*
FixedMallocPool::allocInPool(Pool& p) noexcept
{
  if (!p.num_avail) return nullptr;

  for (unsigned int int_index = 0; int_index < m_avail_length; ++int_index) {
    // Return the index of the first 1 bit
    const int bit_index = find_first_set(p.avail[int_index]) - 1;
    if (bit_index >= 0) {
      const size_t index = int_index * bits_per_int + bit_index;
      if (index < m_obj_per_pool) {
        // Flip bit 1 -> 0
        p.avail[int_index] ^= 1 << bit_index;
        p.num_avail--;
        return static_cast<void*>(p.data + m_obj_bytes * index);
      }
    }
  }

  return nullptr;
}

void*
FixedMallocPool::allocate(size_t bytes)
{
  void* ptr = nullptr;

  for (auto it = m_pool.rbegin(); it != m_pool.rend(); ++it) {
    ptr = allocInPool(*it);
    if (ptr) {
      m_current_bytes += m_obj_bytes;
      m_highwatermark = std::max(m_highwatermark, m_current_bytes);
      break;
    }
  }

  if (!ptr) {
    newPool();
    ptr = allocate(bytes);
  }

  UMPIRE_ASSERT(ptr);
  return ptr;
}

void
FixedMallocPool::deallocate(void* ptr)
{
  for (auto& p : m_pool) {
    const char* t_ptr = reinterpret_cast<char*>(ptr);
    const ptrdiff_t offset = t_ptr - p.data;
    if ((offset >= 0) && (offset < static_cast<ptrdiff_t>(m_data_bytes))) {
      const size_t alloc_index = offset / m_obj_bytes;
      const size_t int_index   = alloc_index / bits_per_int;
      const short  bit_index   = alloc_index % bits_per_int;

      UMPIRE_ASSERT(! (p.avail[int_index] & (1 << bit_index)));

        // Flip bit 0 -> 1
      p.avail[int_index] ^= 1 << bit_index;
      p.num_avail++;

      m_current_bytes -= m_obj_bytes;

      return;
    }
  }

  UMPIRE_ERROR("Could not find the pointer to deallocate");
}

size_t
FixedMallocPool::numPools() const noexcept
{
  return m_pool.size();
}

} // end of namespace strategy
} // end of namespace umpire
