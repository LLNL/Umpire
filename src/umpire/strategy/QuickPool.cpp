// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/QuickPool.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/strategy/PoolCoalesceHeuristic.hpp"
#include "umpire/strategy/mixins/AlignedAllocation.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/memory_sanitizers.hpp"

namespace umpire {
namespace strategy {

#if defined(UMPIRE_V1_DELEGATE_TO_V2)

// Delegated implementation: the pooling algorithm/bookkeeping is delegated
// to a v2 strategy::quick_pool<...> composed over a
// strategy::detail::v1_backed_memory bridge wrapping the v1 parent. This is
// the same MINIMAL-diff approach as SizeLimiter/ThreadSafeAllocator (see
// SizeLimiter.cpp): the v1 AllocationStrategy::allocate_internal()/
// deallocate_internal() counter path (m_current_size/m_high_watermark/
// m_allocation_count, inherited from AllocationStrategy and untouched here)
// still runs exactly as before via the outer Allocator.inl call sequence, so
// getCurrentSize()/getHighWatermark() behavior is unchanged. Only the pool's
// own bookkeeping (m_actual_bytes, m_aligned_bytes, block counts, etc.) is
// delegated -- getActualSize() and friends below forward to the v2 pool.
//
// Heuristic bridging: v1's PoolCoalesceHeuristic<QuickPool> is a
// std::function<std::size_t(const QuickPool&)>, but the v2 quick_pool wants
// a pool_coalesce_heuristic<quick_pool<v1_backed_memory>> (a
// std::function<std::size_t(const quick_pool<v1_backed_memory>&)>) -- these
// are different types since the pool type differs. m_should_coalesce below
// still stores the *v1-shaped* heuristic exactly as the caller supplied it
// (this is what the static percent_releasable()-style factories below
// produce, and what they operate against via the getActualSize()/
// getReleasableSize()/etc. getters below, all of which now forward to the
// v2 delegate). To let the v2 pool run this heuristic internally after each
// deallocate(), we wrap it in a lambda that captures `this` (the v1
// QuickPool) and ignores the v2 pool argument it's given, instead invoking
// `m_should_coalesce(*this)`. This wrapper is passed explicitly as the
// v2 pool constructor's `should_coalesce` argument so the v2 default-arg
// heuristic (percent_releasable_hwm(100) in v2's own shape) never fires.
// There is no infinite recursion: `m_should_coalesce` (the user-supplied v1
// heuristic) only calls the plain getter methods below, never the wrapper
// itself.
QuickPool::QuickPool(const std::string& name, int id, Allocator allocator,
                     const std::size_t first_minimum_pool_allocation_size,
                     const std::size_t next_minimum_pool_allocation_size, std::size_t alignment,
                     PoolCoalesceHeuristic<QuickPool> should_coalesce) noexcept
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "QuickPool"},
      mixins::AlignedAllocation{alignment, allocator.getAllocationStrategy()},
      m_should_coalesce{should_coalesce},
      m_first_minimum_pool_allocation_size{first_minimum_pool_allocation_size},
      m_next_minimum_pool_allocation_size{next_minimum_pool_allocation_size},
      m_v1_backed_parent{std::make_unique<detail::v1_backed_memory>(m_allocator)},
      m_delegate{std::make_unique<quick_pool<detail::v1_backed_memory>>(
          name, m_v1_backed_parent.get(), first_minimum_pool_allocation_size, next_minimum_pool_allocation_size,
          alignment,
          [this](const quick_pool<detail::v1_backed_memory>&) -> std::size_t { return m_should_coalesce(*this); })}
{
  UMPIRE_LOG(Debug, " ( "
                        << "name=\"" << name << "\""
                        << ", id=" << id << ", allocator=\"" << allocator.getName() << "\""
                        << ", first_minimum_pool_allocation_size=" << m_first_minimum_pool_allocation_size
                        << ", next_minimum_pool_allocation_size=" << m_next_minimum_pool_allocation_size
                        << ", alignment=" << alignment << " )");
}

QuickPool::~QuickPool()
{
  UMPIRE_LOG(Debug, "Releasing free blocks to device");
  // Intentionally does NOT call release() here (unlike the non-delegated
  // path below): m_delegate's own destructor performs the equivalent of
  // { m_is_destructing = true; release(); } against its own
  // is-destructing flag when the unique_ptr is torn down as this object's
  // members are destroyed. Calling release() explicitly here would run
  // before that flag is set on the v2 pool, so a backend-shutdown error
  // during aligned_deallocate would propagate out of this destructor
  // instead of being safely swallowed.
}

void* QuickPool::allocate(std::size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");
  void* ret = m_delegate->allocate(bytes);
  UMPIRE_UNPOISON_MEMORY_REGION(m_allocator, ret, bytes);
  return ret;
}

void QuickPool::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  m_delegate->deallocate(ptr);
}

void QuickPool::release()
{
  UMPIRE_LOG(Debug, "()");
  m_delegate->release();
}

std::size_t QuickPool::getReleasableBlocks() const noexcept
{
  return m_delegate->get_releasable_blocks();
}

std::size_t QuickPool::getTotalBlocks() const noexcept
{
  return m_delegate->get_total_blocks();
}

std::size_t QuickPool::getActualSize() const noexcept
{
  return m_delegate->get_actual_size();
}

std::size_t QuickPool::getReleasableSize() const noexcept
{
  return m_delegate->get_releasable_size();
}

std::size_t QuickPool::getActualHighwaterMark() const noexcept
{
  return m_delegate->get_actual_highwatermark();
}

std::size_t QuickPool::getAlignedSize() const noexcept
{
  return m_delegate->get_aligned_size();
}

std::size_t QuickPool::getAlignedHighwaterMark() const noexcept
{
  return m_delegate->get_aligned_highwatermark();
}

Platform QuickPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits QuickPool::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

bool QuickPool::tracksMemoryUse() const noexcept
{
  return false;
}

std::size_t QuickPool::getBlocksInPool() const noexcept
{
  return m_delegate->get_blocks_in_pool();
}

std::size_t QuickPool::getLargestAvailableBlock() noexcept
{
  return m_delegate->get_largest_available_block();
}

void QuickPool::coalesce() noexcept
{
  UMPIRE_LOG(Debug, "()");

  // Emitted natively (not by m_delegate->coalesce(), which would emit its
  // own identical event) so that exactly one "coalesce" event is recorded
  // per call, matching the non-delegated path below and the replay tests
  // that assert on it.
  umpire::event::record([&](auto& event) {
    event.name("coalesce").category(event::category::operation).tag("allocator_name", getName()).tag("replay", "true");
  });

  std::size_t suggested_size{m_should_coalesce(*this)};
  if (0 != suggested_size) {
    UMPIRE_LOG(Debug, "coalesce heuristic true, performing coalesce, suggested size is " << suggested_size);
    do_coalesce(suggested_size);
  }
}

void QuickPool::do_coalesce(std::size_t suggested_size) noexcept
{
  m_delegate->do_coalesce(suggested_size);
}

PoolCoalesceHeuristic<QuickPool> QuickPool::blocks_releasable(std::size_t nblocks)
{
  return
      [=](const strategy::QuickPool& pool) { return pool.getReleasableBlocks() >= nblocks ? pool.getActualSize() : 0; };
}

PoolCoalesceHeuristic<QuickPool> QuickPool::blocks_releasable_hwm(std::size_t nblocks)
{
  return [=](const strategy::QuickPool& pool) {
    return pool.getReleasableBlocks() >= nblocks ? pool.getAlignedHighwaterMark() : 0;
  };
}

PoolCoalesceHeuristic<QuickPool> QuickPool::percent_releasable(int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("Invalid percentage: {}, percentage must be an integer between 0 and 100", percentage));
  }
  if (percentage == 0) {
    return [=](const QuickPool& UMPIRE_UNUSED_ARG(pool)) { return 0; };
  } else if (percentage == 100) {
    return [=](const strategy::QuickPool& pool) {
      return pool.getActualSize() == pool.getReleasableSize() ? pool.getActualSize() : 0;
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);
    return [=](const strategy::QuickPool& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
      return pool.getReleasableSize() >= threshold ? pool.getActualSize() : 0;
    };
  }
}

PoolCoalesceHeuristic<QuickPool> QuickPool::percent_releasable_hwm(int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("Invalid percentage: {}, percentage must be an integer between 0 and 100", percentage));
  }
  if (percentage == 0) {
    return [=](const QuickPool& UMPIRE_UNUSED_ARG(pool)) { return 0; };
  } else if (percentage == 100) {
    return [=](const strategy::QuickPool& pool) {
      return pool.getActualSize() == pool.getReleasableSize() ? pool.getAlignedHighwaterMark() : 0;
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);
    return [=](const strategy::QuickPool& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
      return pool.getReleasableSize() >= threshold ? pool.getAlignedHighwaterMark() : 0;
    };
  }
}

std::ostream& operator<<(std::ostream& out, umpire::strategy::PoolCoalesceHeuristic<QuickPool>&)
{
  return out;
}

#else // !defined(UMPIRE_V1_DELEGATE_TO_V2)

QuickPool::QuickPool(const std::string& name, int id, Allocator allocator,
                     const std::size_t first_minimum_pool_allocation_size,
                     const std::size_t next_minimum_pool_allocation_size, std::size_t alignment,
                     PoolCoalesceHeuristic<QuickPool> should_coalesce) noexcept
    : AllocationStrategy{name, id, allocator.getAllocationStrategy(), "QuickPool"},
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

QuickPool::~QuickPool()
{
  UMPIRE_LOG(Debug, "Releasing free blocks to device");
  m_is_destructing = true;
  release();
}

void* QuickPool::allocate(std::size_t bytes)
{
  UMPIRE_LOG(Debug, "(bytes=" << bytes << ")");
  const std::size_t rounded_bytes{aligned_round_up(bytes)};
  const auto& best = m_size_map.lower_bound(rounded_bytes);

  Chunk* chunk{nullptr};

  if (best == m_size_map.end()) {
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
    chunk = new (chunk_storage) Chunk{ret, size, size};
  } else {
    chunk = (*best).second;
    m_size_map.erase(best);
  }

  UMPIRE_LOG(Debug, "Using chunk " << chunk << " with data " << chunk->data << " and size " << chunk->size
                                   << " for allocation of size " << rounded_bytes);

  if ((chunk->size == chunk->chunk_size) && chunk->free) {
    m_releasable_bytes -= chunk->chunk_size;
    m_releasable_blocks--;
  }

  void* ret = chunk->data;
  m_pointer_map.insert(std::make_pair(ret, chunk));

  chunk->free = false;

  if (rounded_bytes != chunk->size) {
    std::size_t remaining{chunk->size - rounded_bytes};
    UMPIRE_LOG(Debug, "Splitting chunk " << chunk->size << "into " << rounded_bytes << " and " << remaining);

    void* chunk_storage{m_chunk_pool.allocate()};
    Chunk* split_chunk{new (chunk_storage)
                           Chunk{static_cast<char*>(ret) + rounded_bytes, remaining, chunk->chunk_size}};

    auto old_next = chunk->next;
    chunk->next = split_chunk;
    split_chunk->prev = chunk;
    split_chunk->next = old_next;

    if (split_chunk->next)
      split_chunk->next->prev = split_chunk;

    chunk->size = rounded_bytes;
    split_chunk->size_map_it = m_size_map.insert(std::make_pair(remaining, split_chunk));
  }

  m_aligned_bytes += rounded_bytes;
  if (m_aligned_bytes > m_aligned_highwatermark) {
    m_aligned_highwatermark = m_aligned_bytes;
  }

  UMPIRE_UNPOISON_MEMORY_REGION(m_allocator, ret, bytes);
  return ret;
}

void QuickPool::deallocate(void* ptr, std::size_t UMPIRE_UNUSED_ARG(size))
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  auto chunk = (*m_pointer_map.find(ptr)).second;
  chunk->free = true;

  m_aligned_bytes -= chunk->size;

  UMPIRE_LOG(Debug, "Deallocating data held by " << chunk);

  UMPIRE_POISON_MEMORY_REGION(m_allocator, ptr, chunk->size);

  if (chunk->prev && chunk->prev->free == true) {
    auto prev = chunk->prev;
    UMPIRE_LOG(Debug, "Removing chunk " << prev << " from size map");

    m_size_map.erase(prev->size_map_it);

    prev->size += chunk->size;
    prev->next = chunk->next;

    if (prev->next)
      prev->next->prev = prev;

    UMPIRE_LOG(Debug, "Merging with prev " << prev << " and " << chunk);
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

    UMPIRE_LOG(Debug, "Merging with next " << chunk << " and " << next);
    UMPIRE_LOG(Debug, "New size: " << chunk->size);

    UMPIRE_LOG(Debug, "Removing chunk " << next << " from size map");
    m_size_map.erase(next->size_map_it);

    m_chunk_pool.deallocate(next);
  }

  UMPIRE_LOG(Debug, "Inserting chunk " << chunk << " with size " << chunk->size);

  if (chunk->size == chunk->chunk_size) {
    m_releasable_blocks++;
    m_releasable_bytes += chunk->chunk_size;
  }

  chunk->size_map_it = m_size_map.insert(std::make_pair(chunk->size, chunk));
  // can do this with iterator?
  m_pointer_map.erase(ptr);

  std::size_t suggested_size{m_should_coalesce(*this)};
  if (0 != suggested_size) {
    UMPIRE_LOG(Debug, "coalesce heuristic true, performing coalesce.");
    do_coalesce(suggested_size);
  }
}

void QuickPool::release()
{
  UMPIRE_LOG(Debug, "() " << m_size_map.size() << " chunks in free map, m_is_destructing set to " << m_is_destructing);

#if defined(UMPIRE_ENABLE_BACKTRACE)
  std::size_t prev_size{m_actual_bytes};
#endif

  for (auto pair = m_size_map.begin(); pair != m_size_map.end();) {
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
      pair = m_size_map.erase(pair);
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

std::size_t QuickPool::getReleasableBlocks() const noexcept
{
  return m_releasable_blocks;
}

std::size_t QuickPool::getTotalBlocks() const noexcept
{
  return m_total_blocks;
}

std::size_t QuickPool::getActualSize() const noexcept
{
  return m_actual_bytes;
}

std::size_t QuickPool::getReleasableSize() const noexcept
{
  return m_releasable_bytes;
}

std::size_t QuickPool::getActualHighwaterMark() const noexcept
{
  return m_actual_highwatermark;
}

std::size_t QuickPool::getAlignedSize() const noexcept
{
  return m_aligned_bytes;
}

std::size_t QuickPool::getAlignedHighwaterMark() const noexcept
{
  return m_aligned_highwatermark;
}

Platform QuickPool::getPlatform() noexcept
{
  return m_allocator->getPlatform();
}

MemoryResourceTraits QuickPool::getTraits() const noexcept
{
  return m_allocator->getTraits();
}

bool QuickPool::tracksMemoryUse() const noexcept
{
  return false;
}

std::size_t QuickPool::getBlocksInPool() const noexcept
{
  return m_pointer_map.size() + m_size_map.size();
}

std::size_t QuickPool::getLargestAvailableBlock() noexcept
{
  if (!m_size_map.size()) {
    return 0;
  }
  return m_size_map.rbegin()->first;
}

void QuickPool::coalesce() noexcept
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

void QuickPool::do_coalesce(std::size_t suggested_size) noexcept
{
  if (m_size_map.size() > 1) {
    UMPIRE_LOG(Debug, "()");
    release();
    std::size_t size_post{getActualSize()};

    if (size_post < suggested_size) {
      std::size_t alloc_size{suggested_size - size_post};

      UMPIRE_LOG(Debug, "coalescing " << alloc_size << " bytes.");
      auto ptr = allocate(alloc_size);
      deallocate(ptr, alloc_size);
    }
  }
}

PoolCoalesceHeuristic<QuickPool> QuickPool::blocks_releasable(std::size_t nblocks)
{
  return
      [=](const strategy::QuickPool& pool) { return pool.getReleasableBlocks() >= nblocks ? pool.getActualSize() : 0; };
}

PoolCoalesceHeuristic<QuickPool> QuickPool::blocks_releasable_hwm(std::size_t nblocks)
{
  return [=](const strategy::QuickPool& pool) {
    return pool.getReleasableBlocks() >= nblocks ? pool.getAlignedHighwaterMark() : 0;
  };
}

PoolCoalesceHeuristic<QuickPool> QuickPool::percent_releasable(int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("Invalid percentage: {}, percentage must be an integer between 0 and 100", percentage));
  }
  if (percentage == 0) {
    return [=](const QuickPool& UMPIRE_UNUSED_ARG(pool)) { return 0; };
  } else if (percentage == 100) {
    return [=](const strategy::QuickPool& pool) {
      return pool.getActualSize() == pool.getReleasableSize() ? pool.getActualSize() : 0;
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);
    return [=](const strategy::QuickPool& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
      return pool.getReleasableSize() >= threshold ? pool.getActualSize() : 0;
    };
  }
}

PoolCoalesceHeuristic<QuickPool> QuickPool::percent_releasable_hwm(int percentage)
{
  if (percentage < 0 || percentage > 100) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("Invalid percentage: {}, percentage must be an integer between 0 and 100", percentage));
  }
  if (percentage == 0) {
    return [=](const QuickPool& UMPIRE_UNUSED_ARG(pool)) { return 0; };
  } else if (percentage == 100) {
    return [=](const strategy::QuickPool& pool) {
      return pool.getActualSize() == pool.getReleasableSize() ? pool.getAlignedHighwaterMark() : 0;
    };
  } else {
    float f = (float)((float)percentage / (float)100.0);
    return [=](const strategy::QuickPool& pool) {
      // Calculate threshold in bytes from the percentage
      const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
      return pool.getReleasableSize() >= threshold ? pool.getAlignedHighwaterMark() : 0;
    };
  }
}

std::ostream& operator<<(std::ostream& out, umpire::strategy::PoolCoalesceHeuristic<QuickPool>&)
{
  return out;
}

#endif // UMPIRE_V1_DELEGATE_TO_V2

} // end of namespace strategy
} // end namespace umpire
