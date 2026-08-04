//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_quick_pool_HPP
#define UMPIRE_strategy_quick_pool_HPP

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "umpire/error.hpp"
#include "umpire/event/event.hpp"
#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/strategy/mixins/aligned_allocation.hpp"
#include "umpire/strategy/pool_coalesce_heuristic.hpp"
#include "umpire/util/FixedMallocPool.hpp"

#include "fmt/format.h"

namespace umpire {
namespace strategy {

namespace detail {

// SFINAE helper mirroring strategy::detail::thread_safe_platform (see
// thread_safe.hpp): tolerates a `Memory` type without a `platform` member
// alias (e.g. a runtime-typed bridge such as
// strategy::detail::v1_backed_memory), defaulting to `void` instead of a
// hard compile error.
template <typename Memory, typename = void>
struct quick_pool_platform {
  using type = void;
};

template <typename Memory>
struct quick_pool_platform<Memory, std::void_t<typename Memory::platform>> {
  using type = typename Memory::platform;
};

} // namespace detail

//! @brief Best-fit chunk pool with heuristic-driven coalescing
//!
//! This is the API v2 port of the v1 QuickPool algorithm and is
//! behavior-identical to it: allocations are served best-fit from a
//! doubly-linked list of chunks indexed by size, chunks are split on
//! allocation and merged with free neighbors on deallocation, and fully-free
//! blocks are returned to the parent when the configured coalesce heuristic
//! fires or release() is called.
//!
//! For the power-of-2 segregated-bin allocator previously published under
//! this name, see strategy::binned_pool.
//!
//! @par Statistics
//! - get_current_size()/get_highwatermark(): live and peak requested bytes
//! - get_actual_size(): bytes currently obtained from the parent
//! - get_aligned_size()/get_aligned_highwatermark(): alignment-rounded live
//!   and peak bytes, used by the *_hwm coalesce heuristics
//!
//! @tparam Memory The memory source type to wrap (must inherit from memory)
template <typename Memory>
class quick_pool : public allocation_strategy, private mixins::aligned_allocation {
public:
  //! @brief Platform type propagated from wrapped memory source when available
  using platform = typename detail::quick_pool_platform<Memory>::type;

  static constexpr std::size_t s_default_first_block_size{512 * 1024 * 1024};
  static constexpr std::size_t s_default_next_block_size{1 * 1024 * 1024};
  static constexpr std::size_t s_default_alignment{16};

  //! @brief Coalesce when at least `percentage` percent of the pool's actual
  //!        size is releasable; suggested size is the actual size.
  static pool_coalesce_heuristic<quick_pool> percent_releasable(int percentage)
  {
    validate_percentage(percentage);
    if (percentage == 0) {
      return [](const quick_pool&) { return static_cast<std::size_t>(0); };
    } else if (percentage == 100) {
      return [](const quick_pool& pool) {
        return pool.get_actual_size() == pool.get_releasable_size() ? pool.get_actual_size()
                                                                    : static_cast<std::size_t>(0);
      };
    }
    const float fraction = static_cast<float>(percentage) / 100.0f;
    return [fraction](const quick_pool& pool) {
      const std::size_t threshold = static_cast<std::size_t>(fraction * pool.get_actual_size());
      return pool.get_releasable_size() >= threshold ? pool.get_actual_size() : static_cast<std::size_t>(0);
    };
  }

  //! @brief Like percent_releasable(), but the suggested size is the aligned
  //!        high watermark rather than the current actual size.
  static pool_coalesce_heuristic<quick_pool> percent_releasable_hwm(int percentage)
  {
    validate_percentage(percentage);
    if (percentage == 0) {
      return [](const quick_pool&) { return static_cast<std::size_t>(0); };
    } else if (percentage == 100) {
      return [](const quick_pool& pool) {
        return pool.get_actual_size() == pool.get_releasable_size() ? pool.get_aligned_highwatermark()
                                                                    : static_cast<std::size_t>(0);
      };
    }
    const float fraction = static_cast<float>(percentage) / 100.0f;
    return [fraction](const quick_pool& pool) {
      const std::size_t threshold = static_cast<std::size_t>(fraction * pool.get_actual_size());
      return pool.get_releasable_size() >= threshold ? pool.get_aligned_highwatermark() : static_cast<std::size_t>(0);
    };
  }

  //! @brief Coalesce when at least `nblocks` whole blocks are releasable;
  //!        suggested size is the actual size.
  static pool_coalesce_heuristic<quick_pool> blocks_releasable(std::size_t nblocks)
  {
    return [nblocks](const quick_pool& pool) {
      return pool.get_releasable_blocks() >= nblocks ? pool.get_actual_size() : static_cast<std::size_t>(0);
    };
  }

  //! @brief Like blocks_releasable(), but the suggested size is the aligned
  //!        high watermark.
  static pool_coalesce_heuristic<quick_pool> blocks_releasable_hwm(std::size_t nblocks)
  {
    return [nblocks](const quick_pool& pool) {
      return pool.get_releasable_blocks() >= nblocks ? pool.get_aligned_highwatermark() : static_cast<std::size_t>(0);
    };
  }

  //! @brief Construct a new quick_pool.
  //!
  //! @param name Name for this pool instance
  //! @param parent The memory source providing the pool's backing blocks
  //! @param first_minimum_pool_allocation_size Size of the initial block
  //! @param next_minimum_pool_allocation_size Minimum size of subsequent blocks
  //! @param alignment Allocation alignment in bytes (power of 2)
  //! @param should_coalesce Heuristic controlling automatic coalescing
  explicit quick_pool(const std::string& name, Memory* parent,
                      std::size_t first_minimum_pool_allocation_size = s_default_first_block_size,
                      std::size_t next_minimum_pool_allocation_size = s_default_next_block_size,
                      std::size_t alignment = s_default_alignment,
                      pool_coalesce_heuristic<quick_pool> should_coalesce = percent_releasable_hwm(100))
    : allocation_strategy(name, parent),
      mixins::aligned_allocation(alignment, parent),
      m_should_coalesce{should_coalesce},
      m_first_minimum_pool_allocation_size{first_minimum_pool_allocation_size},
      m_next_minimum_pool_allocation_size{next_minimum_pool_allocation_size}
  {
  }

  ~quick_pool() override
  {
    m_is_destructing = true;
    release();
  }

  quick_pool(const quick_pool&) = delete;

  //! @brief Allocate `bytes` from the best-fitting free chunk, growing the
  //!        pool from the parent when no chunk fits.
  void* allocate(std::size_t bytes) override
  {
    // Requested-byte statistics mirror v1's AllocationStrategy::allocate_internal:
    // they are updated only on the public entry points, never by internal
    // coalesce traffic.
    update_current_size(static_cast<std::ptrdiff_t>(bytes));
    return core_allocate(bytes);
  }

  //! @brief Return `ptr` to the pool, merging with free neighbor chunks and
  //!        running the coalesce heuristic.
  void deallocate(void* ptr) override
  {
    auto it = m_pointer_map.find(ptr);
    if (it == m_pointer_map.end()) {
      throw umpire::unknown_allocation(
          fmt::format("quick_pool \"{}\": pointer {} was not allocated by this pool", get_name(), ptr));
    }
    update_current_size(-static_cast<std::ptrdiff_t>(it->second->requested_size));
    core_deallocate(ptr);
  }

  //! @brief Return all fully-free blocks to the parent memory source.
  void release()
  {
    for (auto pair = m_size_map.begin(); pair != m_size_map.end();) {
      chunk* c = pair->second;
      if ((c->size == c->chunk_size) && c->free) {
        m_actual_bytes -= c->chunk_size;
        m_releasable_bytes -= c->chunk_size;
        m_releasable_blocks--;
        m_total_blocks--;
        update_actual_size(-static_cast<std::ptrdiff_t>(c->chunk_size));

        try {
          aligned_deallocate(c->data);
        } catch (...) {
          if (!m_is_destructing) {
            throw;
          }
          // Ignore errors during destruction: the underlying backend may
          // already have shut down.
        }

        m_chunk_pool.deallocate(c);
        pair = m_size_map.erase(pair);
      } else {
        ++pair;
      }
    }
  }

  //! @brief Run the coalesce heuristic and coalesce if it fires.
  void coalesce()
  {
    umpire::event::record([&](auto& event) {
      event.name("coalesce").category(event::category::operation).tag("allocator_name", get_name()).tag("replay",
                                                                                                        "true");
    });

    const std::size_t suggested_size{m_should_coalesce(*this)};
    if (0 != suggested_size) {
      do_coalesce(suggested_size);
    }
  }

  //! @brief Release free blocks and re-allocate a single block sized to
  //!        `suggested_size` so future requests are served contiguously.
  //!
  //! Internal traffic deliberately bypasses the requested-byte statistics,
  //! matching the v1 QuickPool, whose coalesce path calls the raw
  //! allocate/deallocate mechanics rather than the tracked entry points.
  void do_coalesce(std::size_t suggested_size)
  {
    if (m_size_map.size() > 1) {
      release();
      const std::size_t size_post{m_actual_bytes};

      if (size_post < suggested_size) {
        const std::size_t alloc_size{suggested_size - size_post};
        void* ptr = core_allocate(alloc_size);
        core_deallocate(ptr);
      }
    }
  }

private:
  //! Best-fit allocation mechanics; updates aligned/actual statistics but
  //! not the requested-byte counters (see allocate()).
  void* core_allocate(std::size_t bytes)
  {
    const std::size_t rounded_bytes{aligned_round_up(bytes)};
    const auto& best = m_size_map.lower_bound(rounded_bytes);

    chunk* c{nullptr};

    if (best == m_size_map.end()) {
      const std::size_t bytes_to_use{(m_actual_bytes == 0) ? m_first_minimum_pool_allocation_size
                                                           : m_next_minimum_pool_allocation_size};
      const std::size_t size{(rounded_bytes > bytes_to_use) ? rounded_bytes : bytes_to_use};

      void* ret{nullptr};
      try {
        ret = aligned_allocate(size);
      } catch (...) {
        // Give back all fully-free blocks and retry once before giving up.
        release();
        ret = aligned_allocate(size);
      }

      m_actual_bytes += size;
      m_releasable_bytes += size;
      m_releasable_blocks++;
      m_total_blocks++;
      m_actual_highwatermark = (m_actual_bytes > m_actual_highwatermark) ? m_actual_bytes : m_actual_highwatermark;
      update_actual_size(static_cast<std::ptrdiff_t>(size));

      void* chunk_storage{m_chunk_pool.allocate()};
      c = new (chunk_storage) chunk{ret, size, size};
    } else {
      c = best->second;
      m_size_map.erase(best);
    }

    if ((c->size == c->chunk_size) && c->free) {
      m_releasable_bytes -= c->chunk_size;
      m_releasable_blocks--;
    }

    void* ret = c->data;
    m_pointer_map.insert(std::make_pair(ret, c));

    c->free = false;
    c->requested_size = bytes;

    if (rounded_bytes != c->size) {
      const std::size_t remaining{c->size - rounded_bytes};

      void* chunk_storage{m_chunk_pool.allocate()};
      chunk* split_chunk{new (chunk_storage) chunk{static_cast<char*>(ret) + rounded_bytes, remaining, c->chunk_size}};

      auto old_next = c->next;
      c->next = split_chunk;
      split_chunk->prev = c;
      split_chunk->next = old_next;

      if (split_chunk->next) {
        split_chunk->next->prev = split_chunk;
      }

      c->size = rounded_bytes;
      split_chunk->size_map_it = m_size_map.insert(std::make_pair(remaining, split_chunk));
    }

    m_aligned_bytes += rounded_bytes;
    if (m_aligned_bytes > m_aligned_highwatermark) {
      m_aligned_highwatermark = m_aligned_bytes;
    }

    return ret;
  }

  //! Chunk-merge deallocation mechanics; updates aligned statistics but not
  //! the requested-byte counters (see deallocate()).
  void core_deallocate(void* ptr)
  {
    chunk* c = m_pointer_map.find(ptr)->second;
    c->free = true;

    m_aligned_bytes -= c->size;

    if (c->prev && c->prev->free) {
      chunk* prev = c->prev;
      m_size_map.erase(prev->size_map_it);

      prev->size += c->size;
      prev->next = c->next;

      if (prev->next) {
        prev->next->prev = prev;
      }

      m_chunk_pool.deallocate(c);
      c = prev;
    }

    if (c->next && c->next->free) {
      chunk* next = c->next;
      c->size += next->size;
      c->next = next->next;
      if (c->next) {
        c->next->prev = c;
      }

      m_size_map.erase(next->size_map_it);
      m_chunk_pool.deallocate(next);
    }

    if (c->size == c->chunk_size) {
      m_releasable_blocks++;
      m_releasable_bytes += c->chunk_size;
    }

    c->size_map_it = m_size_map.insert(std::make_pair(c->size, c));
    m_pointer_map.erase(ptr);

    const std::size_t suggested_size{m_should_coalesce(*this)};
    if (0 != suggested_size) {
      do_coalesce(suggested_size);
    }
  }

public:
  //! @brief Bytes in fully-free blocks that release() would return to the parent.
  std::size_t get_releasable_size() const noexcept
  {
    return m_releasable_bytes;
  }

  //! @brief Peak bytes obtained from the parent memory source.
  std::size_t get_actual_highwatermark() const noexcept
  {
    return m_actual_highwatermark;
  }

  //! @brief Alignment-rounded live bytes.
  std::size_t get_aligned_size() const noexcept
  {
    return m_aligned_bytes;
  }

  //! @brief Peak alignment-rounded live bytes.
  std::size_t get_aligned_highwatermark() const noexcept
  {
    return m_aligned_highwatermark;
  }

  //! @brief Number of chunks (in-use plus free) held by the pool.
  std::size_t get_blocks_in_pool() const noexcept
  {
    return m_pointer_map.size() + m_size_map.size();
  }

  //! @brief Largest allocation that can be satisfied without growing the pool.
  std::size_t get_largest_available_block() const noexcept
  {
    if (m_size_map.empty()) {
      return 0;
    }
    return m_size_map.rbegin()->first;
  }

  //! @brief Number of fully-free blocks that release() would return.
  std::size_t get_releasable_blocks() const noexcept
  {
    return m_releasable_blocks;
  }

  //! @brief Total number of blocks obtained from the parent.
  std::size_t get_total_blocks() const noexcept
  {
    return m_total_blocks;
  }

private:
  struct chunk;

  //! Allocator placing size-map nodes in a FixedMallocPool, ported verbatim
  //! from the v1 QuickPool to preserve allocation behavior of the pool's own
  //! metadata.
  template <typename Value>
  class pool_allocator {
  public:
    using value_type = Value;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    pool_allocator() : pool{std::make_shared<util::FixedMallocPool>(sizeof(Value))}
    {
    }

    template <typename U>
    pool_allocator(const pool_allocator<U>& other) : pool{other.pool}
    {
    }

    Value* allocate(std::size_t n)
    {
      return static_cast<Value*>(pool->allocate(n));
    }

    void deallocate(Value* data, std::size_t)
    {
      pool->deallocate(data);
    }

    std::shared_ptr<util::FixedMallocPool> pool;
  };

  using pointer_map = std::unordered_map<void*, chunk*>;
  using size_map =
      std::multimap<std::size_t, chunk*, std::less<std::size_t>, pool_allocator<std::pair<const std::size_t, chunk*>>>;

  struct chunk {
    chunk(void* ptr, std::size_t s, std::size_t cs) : data{ptr}, size{s}, chunk_size{cs}
    {
    }

    void* data{nullptr};
    std::size_t size{0};
    std::size_t chunk_size{0};
    std::size_t requested_size{0};
    bool free{true};
    chunk* prev{nullptr};
    chunk* next{nullptr};
    typename size_map::iterator size_map_it;
  };

  static void validate_percentage(int percentage)
  {
    if (percentage < 0 || percentage > 100) {
      throw umpire::runtime_error(
          fmt::format("Invalid percentage: {}, percentage must be an integer between 0 and 100", percentage));
    }
  }

  pointer_map m_pointer_map{};
  size_map m_size_map{};

  util::FixedMallocPool m_chunk_pool{sizeof(chunk)};

  pool_coalesce_heuristic<quick_pool> m_should_coalesce;

  const std::size_t m_first_minimum_pool_allocation_size;
  const std::size_t m_next_minimum_pool_allocation_size;

  std::size_t m_total_blocks{0};
  std::size_t m_releasable_blocks{0};
  std::size_t m_aligned_bytes{0};
  std::size_t m_aligned_highwatermark{0};
  std::size_t m_actual_bytes{0};
  std::size_t m_releasable_bytes{0};
  std::size_t m_actual_highwatermark{0};
  bool m_is_destructing{false};
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_strategy_quick_pool_HPP
