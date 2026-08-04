//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_dynamic_pool_list_HPP
#define UMPIRE_strategy_dynamic_pool_list_HPP

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

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
struct dynamic_pool_list_platform {
  using type = void;
};

template <typename Memory>
struct dynamic_pool_list_platform<Memory, std::void_t<typename Memory::platform>> {
  using type = typename Memory::platform;
};

} // namespace detail

//! @brief Growable block-list pool with heuristic-driven coalescing
//!
//! This is the API v2 port of the v1 DynamicPoolList/DynamicSizePool
//! algorithm and is behavior-identical to it: allocations are served
//! best-fit from an address-ordered free-block list, blocks are split on
//! allocation and merged with free neighbors when returned, and fully-free
//! blocks are released to the parent when the configured coalesce heuristic
//! fires or release() is called.
//!
//! For the eager, per-deallocation coalescing pool previously published
//! under this name, see strategy::coalescing_pool_list.
//!
//! @par Statistics
//! - get_current_size()/get_highwatermark(): live and peak requested bytes
//! - get_actual_size(): bytes currently obtained from the parent
//! - get_aligned_size()/get_aligned_highwatermark(): alignment-rounded live
//!   and peak bytes, used by the *_hwm coalesce heuristics
//!
//! @tparam Memory The memory source type to wrap (must inherit from memory)
template <typename Memory>
class dynamic_pool_list : public allocation_strategy, private mixins::aligned_allocation {
public:
  //! @brief Platform type propagated from wrapped memory source when available
  using platform = typename detail::dynamic_pool_list_platform<Memory>::type;

  static constexpr std::size_t s_default_first_block_size{512 * 1024 * 1024};
  static constexpr std::size_t s_default_next_block_size{1 * 1024 * 1024};
  static constexpr std::size_t s_default_alignment{16};

  //! @brief Coalesce when at least `percentage` percent of the pool's actual
  //!        size is releasable; suggested size is the actual size.
  static pool_coalesce_heuristic<dynamic_pool_list> percent_releasable(int percentage)
  {
    validate_percentage(percentage);
    if (percentage == 0) {
      return [](const dynamic_pool_list&) { return static_cast<std::size_t>(0); };
    } else if (percentage == 100) {
      return [](const dynamic_pool_list& pool) {
        return pool.get_current_size() == 0 ? pool.get_actual_size() : static_cast<std::size_t>(0);
      };
    }
    const float fraction = static_cast<float>(percentage) / 100.0f;
    return [fraction](const dynamic_pool_list& pool) {
      const std::size_t threshold = static_cast<std::size_t>(fraction * pool.get_actual_size());
      return pool.get_releasable_size() >= threshold ? pool.get_actual_size() : static_cast<std::size_t>(0);
    };
  }

  //! @brief Like percent_releasable(), but the suggested size is the aligned
  //!        high watermark rather than the current actual size.
  static pool_coalesce_heuristic<dynamic_pool_list> percent_releasable_hwm(int percentage)
  {
    validate_percentage(percentage);
    if (percentage == 0) {
      return [](const dynamic_pool_list&) { return static_cast<std::size_t>(0); };
    } else if (percentage == 100) {
      return [](const dynamic_pool_list& pool) {
        return pool.get_current_size() == 0 ? pool.get_aligned_highwatermark() : static_cast<std::size_t>(0);
      };
    }
    const float fraction = static_cast<float>(percentage) / 100.0f;
    return [fraction](const dynamic_pool_list& pool) {
      const std::size_t threshold = static_cast<std::size_t>(fraction * pool.get_actual_size());
      return pool.get_releasable_size() >= threshold ? pool.get_aligned_highwatermark() : static_cast<std::size_t>(0);
    };
  }

  //! @brief Coalesce when at least `nblocks` whole blocks are releasable;
  //!        suggested size is the actual size.
  static pool_coalesce_heuristic<dynamic_pool_list> blocks_releasable(std::size_t nblocks)
  {
    return [nblocks](const dynamic_pool_list& pool) {
      return pool.get_releasable_blocks() >= nblocks ? pool.get_actual_size() : static_cast<std::size_t>(0);
    };
  }

  //! @brief Like blocks_releasable(), but the suggested size is the aligned
  //!        high watermark.
  static pool_coalesce_heuristic<dynamic_pool_list> blocks_releasable_hwm(std::size_t nblocks)
  {
    return [nblocks](const dynamic_pool_list& pool) {
      return pool.get_releasable_blocks() >= nblocks ? pool.get_aligned_highwatermark() : static_cast<std::size_t>(0);
    };
  }

  //! @brief Construct a new dynamic_pool_list.
  //!
  //! @param name Name for this pool instance
  //! @param parent The memory source providing the pool's backing blocks
  //! @param first_minimum_pool_allocation_size Minimum size of the initial block
  //! @param next_minimum_pool_allocation_size Minimum size of subsequent blocks
  //! @param alignment Allocation alignment in bytes (power of 2)
  //! @param should_coalesce Heuristic controlling automatic coalescing
  explicit dynamic_pool_list(const std::string& name, Memory* parent,
                             std::size_t first_minimum_pool_allocation_size = s_default_first_block_size,
                             std::size_t next_minimum_pool_allocation_size = s_default_next_block_size,
                             std::size_t alignment = s_default_alignment,
                             pool_coalesce_heuristic<dynamic_pool_list> should_coalesce = percent_releasable_hwm(100))
    : allocation_strategy(name, parent),
      mixins::aligned_allocation(alignment, parent),
      m_should_coalesce{should_coalesce},
      m_first_minimum_pool_allocation_size{first_minimum_pool_allocation_size},
      m_next_minimum_pool_allocation_size{next_minimum_pool_allocation_size}
  {
  }

  dynamic_pool_list(const dynamic_pool_list&) = delete;

  ~dynamic_pool_list() override
  {
    m_is_destructing = true;
    free_released_blocks();
  }

  //! @brief Allocate `bytes` from the best-fitting free block, growing the
  //!        pool from the parent when no block fits.
  void* allocate(std::size_t bytes) override
  {
    // Requested-byte statistics mirror v1's AllocationStrategy::allocate_internal:
    // they are updated only on the public entry points, never by internal
    // coalesce traffic.
    update_current_size(static_cast<std::ptrdiff_t>(bytes));
    return core_allocate(bytes);
  }

  //! @brief Return `ptr` to the pool, merging with adjacent free blocks and
  //!        running the coalesce heuristic.
  void deallocate(void* ptr) override
  {
    block* curr = m_used_blocks;
    block* prev = nullptr;
    for (; curr && curr->data != ptr; curr = curr->next) {
      prev = curr;
    }
    if (!curr) {
      throw umpire::unknown_allocation(
          fmt::format("dynamic_pool_list \"{}\": pointer {} was not allocated by this pool", get_name(), ptr));
    }

    update_current_size(-static_cast<std::ptrdiff_t>(curr->requested_size));
    core_deallocate(curr, prev);

    // The heuristic runs only on the public entry point, matching v1 where
    // it lives in DynamicPoolList::deallocate and the internal
    // DynamicSizePool coalesce traffic never re-triggers it.
    const std::size_t suggested_size{m_should_coalesce(*this)};
    if (0 != suggested_size) {
      do_coalesce(suggested_size);
    }
  }

  //! @brief Return all fully-free blocks to the parent memory source.
  void release()
  {
    free_released_blocks();
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

  //! @brief Bytes in fully-free blocks that release() would return to the parent.
  std::size_t get_releasable_size() const noexcept
  {
    std::size_t nbytes = 0;
    for (block* temp = m_free_blocks; temp; temp = temp->next) {
      if (temp->size == temp->block_size) {
        nbytes += temp->block_size;
      }
    }
    return nbytes;
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

  //! @brief Number of blocks (in-use plus free) held by the pool.
  std::size_t get_blocks_in_pool() const noexcept
  {
    std::size_t total_blocks{0};
    for (block* curr = m_used_blocks; curr; curr = curr->next) {
      total_blocks += 1;
    }
    for (block* curr = m_free_blocks; curr; curr = curr->next) {
      total_blocks += 1;
    }
    return total_blocks;
  }

  //! @brief Largest allocation that can be satisfied without growing the pool.
  std::size_t get_largest_available_block() const noexcept
  {
    std::size_t largest_block{0};
    for (block* temp = m_free_blocks; temp; temp = temp->next) {
      if (temp->size > largest_block) {
        largest_block = temp->size;
      }
    }
    return largest_block;
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
  struct block {
    char* data{nullptr};
    std::size_t size{0};
    std::size_t block_size{0};
    std::size_t requested_size{0};
    block* next{nullptr};
  };

  //! Best-fit allocation mechanics; updates aligned/actual statistics but
  //! not the requested-byte counters (see allocate()).
  void* core_allocate(std::size_t bytes)
  {
    const std::size_t rounded_bytes{aligned_round_up(bytes)};
    block* best{nullptr};
    block* prev{nullptr};

    find_usable_block(best, prev, rounded_bytes);

    if (!best) {
      allocate_block(best, prev, rounded_bytes);
    }

    split_block(best, prev, rounded_bytes);

    // Push node to the front of the used-block list.
    best->next = m_used_blocks;
    m_used_blocks = best;
    best->requested_size = bytes;

    m_aligned_bytes += rounded_bytes;
    if (m_aligned_bytes > m_aligned_highwatermark) {
      m_aligned_highwatermark = m_aligned_bytes;
    }

    return m_used_blocks->data;
  }

  //! Block-merge deallocation mechanics; updates aligned statistics but not
  //! the requested-byte counters (see deallocate()).
  void core_deallocate(block* curr, block* prev)
  {
    m_aligned_bytes -= curr->size;
    release_block(curr, prev);
  }

  static void validate_percentage(int percentage)
  {
    if (percentage < 0 || percentage > 100) {
      throw umpire::runtime_error(
          fmt::format("Invalid percentage {}, percentage must be an integer between 0 and 100", percentage));
    }
  }

  //! Best-fit search over the free-block list.
  void find_usable_block(block*& best, block*& prev, std::size_t size)
  {
    best = prev = nullptr;
    for (block *iter = m_free_blocks, *iter_prev = nullptr; iter; iter = iter->next) {
      if (iter->size >= size && (!best || iter->size < best->size)) {
        best = iter;
        prev = iter_prev;
        if (iter->size == size) {
          break; // Exact match, look no further.
        }
      }
      iter_prev = iter;
    }
  }

  //! Grow the pool with a new block from the parent memory source.
  void allocate_block(block*& curr, block*& prev, std::size_t size)
  {
    if (m_free_blocks == nullptr && m_used_blocks == nullptr) {
      size = std::max(size, m_first_minimum_pool_allocation_size);
    } else {
      size = std::max(size, m_next_minimum_pool_allocation_size);
    }

    curr = nullptr;
    prev = nullptr;
    void* data{nullptr};

    try {
      data = aligned_allocate(size);
    } catch (...) {
      // Give back all fully-free blocks and retry once before giving up.
      free_released_blocks();
      data = aligned_allocate(size);
    }

    m_actual_bytes += size;
    m_actual_highwatermark = (m_actual_bytes > m_actual_highwatermark) ? m_actual_bytes : m_actual_highwatermark;
    m_releasable_blocks++;
    m_total_blocks++;
    update_actual_size(static_cast<std::ptrdiff_t>(size));

    curr = static_cast<block*>(m_block_pool.allocate());

    // Find next and prev such that next->data is still smaller than data
    // (keep the free list address-ordered).
    block* next;
    for (next = m_free_blocks; next && next->data < data; next = next->next) {
      prev = next;
    }

    curr->data = static_cast<char*>(data);
    curr->size = size;
    curr->block_size = size;
    curr->requested_size = 0;
    curr->next = next;

    if (prev) {
      prev->next = curr;
    } else {
      m_free_blocks = curr;
    }
  }

  //! Carve `size` bytes off the front of free block `curr`.
  void split_block(block*& curr, block*& prev, const std::size_t size)
  {
    block* next;

    if (curr->size == curr->block_size) {
      m_releasable_blocks--;
    }

    if (curr->size == size) {
      next = curr->next;
    } else {
      const std::size_t remaining = curr->size - size;
      block* new_block = static_cast<block*>(m_block_pool.allocate());
      if (!new_block) {
        return;
      }
      new_block->data = curr->data + size;
      new_block->size = remaining;
      new_block->block_size = 0;
      new_block->requested_size = 0;
      new_block->next = curr->next;
      next = new_block;
      curr->size = size;
    }

    if (prev) {
      prev->next = next;
    } else {
      m_free_blocks = next;
    }
  }

  //! Move `curr` from the used list into the address-ordered free list,
  //! merging with physically adjacent free neighbors.
  void release_block(block* curr, block* prev)
  {
    if (prev) {
      prev->next = curr->next;
    } else {
      m_used_blocks = curr->next;
    }

    // Find insertion point in the address-ordered free list.
    prev = nullptr;
    for (block* temp = m_free_blocks; temp && (temp->data < curr->data); temp = temp->next) {
      prev = temp;
    }

    block* next = prev ? prev->next : m_free_blocks;

    // Merge with prev when physically adjacent and curr is a split remainder.
    if (prev && prev->data + prev->size == curr->data && !curr->block_size) {
      prev->size = prev->size + curr->size;
      m_block_pool.deallocate(curr); // keep data
      curr = prev;
    } else if (prev) {
      prev->next = curr;
    } else {
      m_free_blocks = curr;
    }

    // Merge with next when physically adjacent and next is a split remainder.
    if (next && curr->data + curr->size == next->data && !next->block_size) {
      curr->size = curr->size + next->size;
      curr->next = next->next;
      m_block_pool.deallocate(next); // keep data
    } else {
      curr->next = next;
    }

    if (curr->size == curr->block_size) {
      m_releasable_blocks++;
    }
  }

  //! Return every fully-free block to the parent; returns bytes freed.
  std::size_t free_released_blocks()
  {
    block* curr = m_free_blocks;
    block* prev = nullptr;

    std::size_t freed = 0;

    while (curr) {
      block* next = curr->next;
      // The free-block list may contain partially-released blocks; only
      // whole original blocks can be returned to the parent.
      if (curr->size == curr->block_size) {
        m_actual_bytes -= curr->size;
        m_releasable_blocks--;
        m_total_blocks--;
        update_actual_size(-static_cast<std::ptrdiff_t>(curr->size));

        freed += curr->size;
        try {
          aligned_deallocate(curr->data);
        } catch (...) {
          if (!m_is_destructing) {
            throw;
          }
          // Ignore errors during destruction: the underlying backend may
          // already have shut down.
        }

        if (prev) {
          prev->next = curr->next;
        } else {
          m_free_blocks = curr->next;
        }

        m_block_pool.deallocate(curr);
      } else {
        prev = curr;
      }
      curr = next;
    }

    return freed;
  }

  std::size_t get_free_block_count() const noexcept
  {
    std::size_t nb = 0;
    for (block* temp = m_free_blocks; temp; temp = temp->next) {
      if (temp->size == temp->block_size) {
        nb++;
      }
    }
    return nb;
  }

public:
  //! Release free blocks and re-allocate a single block sized to
  //! `suggested_size` so future requests are served contiguously.
  //!
  //! Internal traffic deliberately bypasses the requested-byte statistics
  //! and the coalesce heuristic, matching v1's DynamicSizePool::coalesce.
  //!
  //! Exposed publicly (mirroring quick_pool<Memory>::do_coalesce(), which is
  //! already public) so that callers wrapping this pool (e.g. a
  //! v1-compatibility bridge that computes the suggested size itself, to
  //! emit its own event exactly once rather than relying on this class's
  //! own coalesce()) can invoke the actual coalesce operation directly.
  void do_coalesce(std::size_t suggested_size)
  {
    if (get_free_block_count() > 1) {
      free_released_blocks();
      const std::size_t size_post{m_actual_bytes};

      if (size_post < suggested_size) {
        const std::size_t alloc_size{suggested_size - size_post};
        void* ptr = core_allocate(alloc_size);

        block* curr = m_used_blocks;
        block* prev = nullptr;
        for (; curr && curr->data != ptr; curr = curr->next) {
          prev = curr;
        }
        if (curr) {
          core_deallocate(curr, prev);
        }
      }
    }
  }

private:
  util::FixedMallocPool m_block_pool{sizeof(block)};

  block* m_used_blocks{nullptr};
  block* m_free_blocks{nullptr};

  pool_coalesce_heuristic<dynamic_pool_list> m_should_coalesce;

  const std::size_t m_first_minimum_pool_allocation_size;
  const std::size_t m_next_minimum_pool_allocation_size;

  std::size_t m_aligned_bytes{0};
  std::size_t m_aligned_highwatermark{0};
  std::size_t m_actual_bytes{0};
  std::size_t m_actual_highwatermark{0};
  std::size_t m_releasable_blocks{0};
  std::size_t m_total_blocks{0};
  bool m_is_destructing{false};
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_strategy_dynamic_pool_list_HPP
