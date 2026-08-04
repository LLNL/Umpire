//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/FixedPool.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "umpire/util/Macros.hpp"
#include "umpire/util/find_first_set.hpp"

#if !defined(_MSC_VER)
#define _XOPEN_SOURCE_EXTENDED 1
#include <strings.h>
#endif

namespace umpire {
namespace strategy {

static constexpr std::size_t bits_per_int = sizeof(int) * 8;

#if defined(UMPIRE_V1_DELEGATE_TO_V2)

FixedPool::Pool::Pool(AllocationStrategy*, const std::size_t, const std::size_t, const std::size_t)
    : strategy(nullptr), data(nullptr), avail(nullptr), num_avail(0)
{
  // Unused in the delegated branch: FixedPool::newPool()/allocInPool() are
  // never called, so this constructor body is unreachable. It exists only
  // because struct Pool must remain a complete type for m_pool's (unused)
  // std::vector<Pool> member declaration to compile identically across both
  // branches.
}

FixedPool::FixedPool(const std::string& name, int id, Allocator allocator, const std::size_t object_bytes,
                     const std::size_t objects_per_pool) noexcept
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "FixedPool"},
      m_strategy{allocator.getAllocationStrategy()},
      m_obj_bytes{object_bytes},
      m_obj_per_pool{objects_per_pool},
      m_data_bytes{m_obj_bytes * m_obj_per_pool},
      m_avail_bytes{objects_per_pool / bits_per_int + 1},
      m_current_bytes{0},
      m_actual_bytes{0},
      m_highwatermark{0},
      m_pool{},
      m_v1_backed_parent{std::make_unique<detail::v1_backed_memory>(m_strategy)},
      m_delegate{std::make_unique<fixed_pool<detail::v1_backed_memory>>(name, m_v1_backed_parent.get(), object_bytes,
                                                                        objects_per_pool)},
      m_native_highwatermark{0}
{
  UMPIRE_LOG(Debug, "(name=\"" << name << "\", id=" << id << ", allocator=\"" << allocator.getName()
                                << "\", object_bytes=" << object_bytes << ", objects_per_pool=" << objects_per_pool
                                << ")");
}

FixedPool::~FixedPool()
{
  // m_delegate's own destructor returns every pool it owns back to the
  // v1-backed parent (see v2 fixed_pool<Memory>::~fixed_pool()); unlike the
  // non-delegated path below, there is no leaked-address diagnostic here
  // since v2's fixed_pool tracks pools generically rather than per-bit
  // availability, and it always releases all pools unconditionally at
  // destruction (matching the "no leaks" branch of v1's destructor).
}

void FixedPool::newPool()
{
  // Unused in the delegated branch; v2's fixed_pool<Memory> grows its own
  // pools internally inside allocate().
}

void* FixedPool::allocInPool(Pool&)
{
  // Unused in the delegated branch.
  return nullptr;
}

void* FixedPool::allocate(std::size_t bytes)
{
  // Check that bytes passed matches m_obj_bytes or bytes was not passed
  // (default = 0), matching v1's tolerant assert-only behavior. v2's
  // fixed_pool<Memory>::allocate() strictly throws std::invalid_argument
  // when size != object_size_ (including for size == 0), so the pool's
  // configured object size is always requested from the delegate here
  // rather than the caller-supplied `bytes`, preserving v1's more permissive
  // contract.
  UMPIRE_ASSERT(!bytes || bytes == m_obj_bytes);

  void* ptr = m_delegate->allocate(m_obj_bytes);

  if (ptr) {
    m_current_bytes += m_obj_bytes;
    m_native_highwatermark = std::max(m_native_highwatermark, m_current_bytes);
  } else {
    UMPIRE_ERROR(runtime_error, fmt::format("FixedPool::allocate(size={}): Could not allocate.", m_obj_bytes));
  }

  return ptr;
}

void FixedPool::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  m_delegate->deallocate(ptr);
  m_current_bytes -= m_obj_bytes;
}

void FixedPool::release()
{
  m_delegate->release();
}

std::size_t FixedPool::getCurrentSize() const noexcept
{
  return m_current_bytes;
}

std::size_t FixedPool::getActualSize() const noexcept
{
  // v2's fixed_pool<Memory> has no bitmap-overhead concept (it uses a
  // std::vector free list rather than v1's malloc'd availability bitmap per
  // pool), so this replicates v1's exact m_actual_bytes formula natively:
  // each pool contributes `m_avail_bytes` (bitmap bytes) + `m_data_bytes`
  // (object storage bytes), summed across `get_pool_count()` pools.
  return m_delegate->get_pool_count() * (m_avail_bytes + m_data_bytes);
}

std::size_t FixedPool::getHighWatermark() const noexcept
{
  // v2's fixed_pool<Memory> exposes no high-watermark getter at all, so the
  // peak is tracked natively (see allocate() above), mirroring v1's running
  // max-of-current-bytes computation exactly.
  return m_native_highwatermark;
}

Platform FixedPool::getPlatform() noexcept
{
  return m_strategy->getPlatform();
}

MemoryResourceTraits FixedPool::getTraits() const noexcept
{
  return m_strategy->getTraits();
}

std::size_t FixedPool::numPools() const noexcept
{
  return m_delegate->get_pool_count();
}

bool FixedPool::pointerIsFromPool(void* ptr) const noexcept
{
  return m_delegate->owns(ptr);
}

#else // !defined(UMPIRE_V1_DELEGATE_TO_V2)

FixedPool::Pool::Pool(AllocationStrategy* allocation_strategy, const std::size_t object_bytes,
                      const std::size_t objects_per_pool, const std::size_t avail_bytes)
    : strategy(allocation_strategy),
      data(reinterpret_cast<char*>(strategy->allocate_internal(object_bytes * objects_per_pool))),
      avail(reinterpret_cast<int*>(std::malloc(avail_bytes))),
      num_avail(objects_per_pool)
{
  // Set all bits to 1
  const unsigned char not_zero = static_cast<unsigned char>(~0);
  std::memset(avail, not_zero, avail_bytes);
}

FixedPool::FixedPool(const std::string& name, int id, Allocator allocator, const std::size_t object_bytes,
                     const std::size_t objects_per_pool) noexcept
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "FixedPool"},
      m_strategy{allocator.getAllocationStrategy()},
      m_obj_bytes{object_bytes},
      m_obj_per_pool{objects_per_pool},
      m_data_bytes{m_obj_bytes * m_obj_per_pool},
      m_avail_bytes{objects_per_pool / bits_per_int + 1},
      m_current_bytes{0},
      m_actual_bytes{0},
      m_highwatermark{0},
      m_pool{}
{
  newPool();
}

FixedPool::~FixedPool()
{
  std::vector<void*> leaked_addrs{};

  for (auto& p : m_pool) {
    if (m_obj_per_pool != p.num_avail) {
      for (unsigned int int_index = 0; int_index < m_avail_bytes; ++int_index)
        for (unsigned int bit_index = 0; bit_index < bits_per_int; ++bit_index) {
          if (!(p.avail[int_index] & 1 << bit_index)) {
            const std::size_t index{int_index * bits_per_int + bit_index};
            leaked_addrs.push_back(static_cast<void*>(p.data + m_obj_bytes * index));
          }
        }
    }
  }

  if (leaked_addrs.size() > 0) {
    const std::size_t max_addr{25};
    std::stringstream ss;
    ss << "There are " << leaked_addrs.size() << " addresses";
    ss << " not deallocated at destruction. This will cause leak(s). ";
    if (leaked_addrs.size() <= max_addr)
      ss << "Addresses:";
    else
      ss << "First " << max_addr << " addresses:";
    for (std::size_t i = 0; i < std::min(max_addr, leaked_addrs.size()); ++i) {
      if (i % 5 == 0)
        ss << "\n\t";
      ss << " " << leaked_addrs[i];
    }
    UMPIRE_LOG(Warning, ss.str());
  } else {
    for (auto& p : m_pool) {
      p.strategy->deallocate_internal(p.data, m_data_bytes);
      std::free(p.avail);
    }
  }
}

void FixedPool::newPool()
{
  m_pool.emplace_back(m_strategy, m_obj_bytes, m_obj_per_pool, m_avail_bytes * sizeof(int));
  m_actual_bytes += m_avail_bytes + m_data_bytes;
}

void* FixedPool::allocInPool(Pool& p)
{
  if (!p.num_avail)
    return nullptr;

  for (unsigned int int_index = 0; int_index < m_avail_bytes; ++int_index) {
    // Return the index of the first 1 bit
    const int bit_index = util::find_first_set(p.avail[int_index]) - 1;
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

void* FixedPool::allocate(std::size_t bytes)
{
  // Check that bytes passed matches m_obj_bytes or bytes was not passed
  // (default = 0)
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
    UMPIRE_ERROR(runtime_error, fmt::format("FixedPool::allocate(size={}): Could not allocate.", m_obj_bytes));
  }
  return ptr;
}

void FixedPool::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  for (auto& p : m_pool) {
    const char* t_ptr = reinterpret_cast<char*>(ptr);
    const ptrdiff_t offset = t_ptr - p.data;
    if ((offset >= 0) && (offset < static_cast<ptrdiff_t>(m_data_bytes))) {
      const std::size_t alloc_index = offset / m_obj_bytes;
      const std::size_t int_index = alloc_index / bits_per_int;
      const short bit_index = alloc_index % bits_per_int;

      UMPIRE_ASSERT(!(p.avail[int_index] & (1 << bit_index)));

      // Flip bit 0 -> 1
      p.avail[int_index] ^= 1 << bit_index;
      p.num_avail++;

      m_current_bytes -= m_obj_bytes;

      return;
    }
  }

  UMPIRE_ERROR(runtime_error, "Could not find the pointer to deallocate");
}

void FixedPool::release()
{
  for (auto& p : m_pool) {
    if (m_obj_per_pool == p.num_avail) {
      p.strategy->deallocate_internal(p.data, m_data_bytes);
      std::free(p.avail);
    }
  }
  m_pool.erase(std::remove_if(m_pool.begin(), m_pool.end(), [&](Pool& p) { return m_obj_per_pool == p.num_avail; }),
               m_pool.end());
}

std::size_t FixedPool::getCurrentSize() const noexcept
{
  return m_current_bytes;
}

std::size_t FixedPool::getActualSize() const noexcept
{
  return m_actual_bytes;
}

std::size_t FixedPool::getHighWatermark() const noexcept
{
  return m_highwatermark;
}

Platform FixedPool::getPlatform() noexcept
{
  return m_strategy->getPlatform();
}

MemoryResourceTraits FixedPool::getTraits() const noexcept
{
  return m_strategy->getTraits();
}

std::size_t FixedPool::numPools() const noexcept
{
  return m_pool.size();
}

bool FixedPool::pointerIsFromPool(void* ptr) const noexcept
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

#endif // defined(UMPIRE_V1_DELEGATE_TO_V2)

} // end of namespace strategy
} // end of namespace umpire
