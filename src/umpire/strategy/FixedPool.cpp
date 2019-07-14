//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/FixedPool.hpp"

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
namespace strategy {

static constexpr std::size_t bits_per_int = sizeof(int) * 8;

FixedPool::Pool::Pool(AllocationStrategy* allocation_strategy,
                      const std::size_t object_bytes, const std::size_t objects_per_pool,
                      const std::size_t avail_bytes) :
  strategy(allocation_strategy),
  data(reinterpret_cast<char*>(strategy->allocate(object_bytes * objects_per_pool))),
  avail(reinterpret_cast<int*>(std::malloc(avail_bytes))),
  num_avail(objects_per_pool)
{
  // Set all bits to 1
  const unsigned char not_zero = static_cast<unsigned char>(~0);
  std::memset(avail, not_zero, avail_bytes);
}

FixedPool::FixedPool(const std::string& name, int id,
                     Allocator allocator, const std::size_t object_bytes,
                     const std::size_t objects_per_pool) noexcept :
  AllocationStrategy(name, id),
  m_strategy(allocator.getAllocationStrategy()),
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

FixedPool::~FixedPool()
{
  for (auto& p : m_pool) {
    if (p.num_avail != m_obj_per_pool) {
      UMPIRE_LOG(Debug, "Did not release all memory");
    }
    p.strategy->deallocate(p.data);
    std::free(p.avail);
  }
}

void
FixedPool::newPool()
{
  m_pool.emplace_back(m_strategy, m_obj_bytes, m_obj_per_pool, m_avail_length * sizeof(int));
}

void*
FixedPool::allocInPool(Pool& p)
{
  if (!p.num_avail) return nullptr;

  for (unsigned int int_index = 0; int_index < m_avail_length; ++int_index) {
    // Return the index of the first 1 bit
    const int bit_index = find_first_set(p.avail[int_index]) - 1;
    if (bit_index >= 0) {
      const std::size_t index = int_index * bits_per_int + bit_index;
      if (index < m_obj_per_pool) {
        // Flip bit 1 -> 0
        p.avail[int_index] ^= 1 << bit_index;
        p.num_avail--;
        return static_cast<void*>(p.data + m_obj_bytes * index);
      }
    }
  }

  UMPIRE_ASSERT("FixedPool::allocInPool(): num_avail > 0, but no available slots" && 0);
  return nullptr;
}

void*
FixedPool::allocate(std::size_t bytes)
{
  // Check that bytes passed matches m_obj_bytes or bytes was not passed (default = 0)
  UMPIRE_ASSERT(!bytes || bytes == m_obj_bytes);

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

  if (!ptr) {
    UMPIRE_ERROR("FixedPool::allocate(size=" << m_obj_bytes << "): Could not allocate");
  }
  return ptr;
}

void
FixedPool::deallocate(void* ptr)
{
  for (auto& p : m_pool) {
    const char* t_ptr = reinterpret_cast<char*>(ptr);
    const ptrdiff_t offset = t_ptr - p.data;
    if ((offset >= 0) && (offset < static_cast<ptrdiff_t>(m_data_bytes))) {
      const std::size_t alloc_index = offset / m_obj_bytes;
      const std::size_t int_index   = alloc_index / bits_per_int;
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

std::size_t
FixedPool::getCurrentSize() const noexcept
{
  return m_current_bytes;
}

std::size_t
FixedPool::getActualSize() const noexcept
{
  const std::size_t avail_bytes = m_obj_per_pool/bits_per_int + 1;
  return m_pool.size() * (m_obj_per_pool * m_obj_bytes + avail_bytes + sizeof(Pool))
    + sizeof(FixedPool);
}

std::size_t
FixedPool::getHighWatermark() const noexcept
{
  return m_highwatermark;
}

Platform
FixedPool::getPlatform() noexcept
{
  return m_strategy->getPlatform();
}

std::size_t
FixedPool::numPools() const noexcept
{
  return m_pool.size();
}

bool 
FixedPool::pointerIsFromPool(void* ptr) const noexcept
{
  for (auto& p : m_pool) {
    const char* t_ptr = reinterpret_cast<char*>(ptr);
    const ptrdiff_t offset = t_ptr - p.data;
    if ((offset >= 0) && (offset < static_cast<ptrdiff_t>(m_data_bytes))) {
      return true;
    }
  }

  return false;
}

} // end of namespace strategy
} // end of namespace umpire
