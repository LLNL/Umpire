//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_dynamic_pool_list_HPP
#define UMPIRE_strategy_dynamic_pool_list_HPP

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/util/error.hpp"

#include "fmt/format.h"

#include <cstddef>
#include <list>
#include <string>

namespace umpire {
namespace strategy {

//! @brief Dynamic memory pool with variable-sized blocks and coalescing
//!
//! The dynamic_pool_list strategy manages variable-sized memory allocations
//! efficiently by maintaining a list of memory blocks and coalescing adjacent
//! free blocks to reduce fragmentation. This is ideal for workloads with
//! variable allocation sizes.
//!
//! @par Block Management
//! - Maintains std::list<block> to track all blocks (free and allocated)
//! - Each block has: pointer, size, and free flag
//! - Metadata stored separately (not inline) for simplicity
//! - Blocks are ordered by address for efficient coalescing
//!
//! @par Allocation Strategy
//! - First-fit algorithm: finds first free block large enough
//! - Splits block if remainder > min_alloc_size
//! - Automatically grows pool if no suitable block found
//! - Growth increases pool size by growth_factor
//!
//! @par Deallocation and Coalescing
//! - Marks block as free on deallocation
//! - Immediately coalesces with adjacent free blocks
//! - Reduces fragmentation over time
//! - Detects double-free and unknown pointer errors
//!
//! @par Memory Release
//! - release() method returns free blocks to parent
//! - Only releases blocks at pool boundaries
//! - Statistics track total, allocated, and free sizes
//!
//! @par Platform Propagation
//! The platform type is propagated from the wrapped memory source:
//! - dynamic_pool_list<host_memory>::platform is host_platform
//! - dynamic_pool_list<cuda_device_memory>::platform is cuda_platform
//!
//! @par Performance
//! - O(n) allocation (first-fit search through block list)
//! - O(n) deallocation (find block + coalesce neighbors)
//! - O(1) statistics queries
//! - Trade-off: slower than fixed_pool but handles variable sizes
//!
//! @par Composition
//! Can wrap any memory source:
//! - Resources: dynamic_pool_list<host_memory>
//! - Thread-safe: thread_safe<dynamic_pool_list<host_memory>>
//!
//! @tparam Memory The memory source type to wrap (must inherit from memory)
template<typename Memory>
class dynamic_pool_list : public allocation_strategy {
public:
  //! @brief Platform type propagated from wrapped memory source
  using platform = typename Memory::platform;

private:
  //! @brief Block metadata for tracking memory regions
  struct block {
    void* ptr;          //!< Pointer to block start
    std::size_t size;   //!< Size of block in bytes
    bool is_free;       //!< True if block is available for allocation

    block(void* p, std::size_t s, bool free)
      : ptr(p), size(s), is_free(free) {}
  };

  std::list<block> blocks_;           //!< List of all blocks (ordered by address)
  std::vector<void*> pools_;          //!< Track pool base pointers for cleanup
  std::size_t initial_pool_size_;     //!< Initial pool allocation size
  std::size_t min_alloc_size_;        //!< Minimum allocation size for splitting
  double growth_factor_;              //!< Pool growth multiplier

  // Statistics
  std::size_t total_size_;            //!< Total memory allocated from parent
  std::size_t allocated_size_;        //!< Currently allocated to users
  std::size_t free_size_;             //!< Currently free (available for allocation)
  std::size_t next_pool_size_;        //!< Size for next pool allocation

  //! @brief Allocate a new pool from parent and add to block list
  //!
  //! @param size Size of pool to allocate
  void allocate_pool(std::size_t size) {
    void* pool = parent_->allocate(size);

    if (!pool) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("dynamic_pool_list: failed to allocate pool of {} bytes", size));
    }

    // Track pool for cleanup
    pools_.push_back(pool);

    // Find insertion point to keep blocks sorted by address
    auto it = blocks_.begin();
    while (it != blocks_.end() && it->ptr < pool) {
      ++it;
    }

    // Insert new free block at correct position
    blocks_.insert(it, block(pool, size, true));

    total_size_ += size;
    free_size_ += size;
  }

  //! @brief Find the block containing the given pointer
  //!
  //! @param ptr Pointer to search for
  //! @return Iterator to block, or blocks_.end() if not found
  typename std::list<block>::iterator find_block(void* ptr) {
    for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
      if (it->ptr == ptr) {
        return it;
      }
    }
    return blocks_.end();
  }

  //! @brief Coalesce adjacent free blocks
  //!
  //! After marking a block as free, this merges it with neighboring free
  //! blocks to reduce fragmentation.
  //!
  //! @param it Iterator to the newly freed block
  void coalesce(typename std::list<block>::iterator it) {
    if (it == blocks_.end()) return;

    // Try to coalesce with next block
    auto next = std::next(it);
    while (next != blocks_.end() && next->is_free) {
      // Check if blocks are contiguous
      char* it_end = static_cast<char*>(it->ptr) + it->size;
      if (it_end == next->ptr) {
        // Merge next block into current
        it->size += next->size;
        blocks_.erase(next);
        next = std::next(it);
      } else {
        break;
      }
    }

    // Try to coalesce with previous block
    if (it != blocks_.begin()) {
      auto prev = std::prev(it);
      if (prev->is_free) {
        char* prev_end = static_cast<char*>(prev->ptr) + prev->size;
        if (prev_end == it->ptr) {
          // Merge current block into previous
          prev->size += it->size;
          blocks_.erase(it);
        }
      }
    }
  }

public:
  //! @brief Construct a dynamic pool with configurable parameters
  //!
  //! @param name Name for this pool instance
  //! @param parent The memory source to wrap (must not be null)
  //! @param initial_pool_size Initial pool size in bytes (default 64KB)
  //! @param min_alloc_size Minimum size for block splitting (default 4KB)
  //! @param growth_factor Pool growth multiplier (default 2.0)
  //!
  //! @throws std::invalid_argument if parent is null (validated by base)
  //! @throws std::invalid_argument if initial_pool_size is 0
  //! @throws std::invalid_argument if min_alloc_size is 0
  //! @throws std::invalid_argument if growth_factor <= 1.0
  explicit dynamic_pool_list(
      const std::string& name,
      Memory* parent,
      std::size_t initial_pool_size = 64 * 1024,
      std::size_t min_alloc_size = 4 * 1024,
      double growth_factor = 2.0)
    : allocation_strategy(name, parent)
    , initial_pool_size_(initial_pool_size)
    , min_alloc_size_(min_alloc_size)
    , growth_factor_(growth_factor)
    , total_size_(0)
    , allocated_size_(0)
    , free_size_(0)
    , next_pool_size_(initial_pool_size)
  {
    if (initial_pool_size_ == 0) {
      UMPIRE_ERROR(std::invalid_argument,
                   "dynamic_pool_list: initial_pool_size must be greater than 0");
    }

    if (min_alloc_size_ == 0) {
      UMPIRE_ERROR(std::invalid_argument,
                   "dynamic_pool_list: min_alloc_size must be greater than 0");
    }

    if (growth_factor_ <= 1.0) {
      UMPIRE_ERROR(std::invalid_argument,
                   "dynamic_pool_list: growth_factor must be greater than 1.0");
    }

    // Allocate initial pool
    allocate_pool(initial_pool_size_);
  }

  //! @brief Destructor - returns all pools to parent
  ~dynamic_pool_list() {
    // Return all pools to parent
    // Note: This assumes all user allocations have been deallocated
    // If there are still allocated blocks, this will leak them
    for (void* pool : pools_) {
      parent_->deallocate(pool);
    }
  }

  //! @brief Allocate memory of specified size
  //!
  //! Uses first-fit algorithm to find a suitable free block.
  //! Splits block if remainder is large enough (> min_alloc_size).
  //! Automatically grows pool if no suitable block found.
  //!
  //! @param size Number of bytes to allocate
  //! @return Pointer to allocated memory
  //! @throws out_of_memory_error if allocation fails
  void* allocate(std::size_t size) override {
    if (size == 0) {
      return nullptr;
    }

    // First-fit: find first free block large enough
    for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
      if (it->is_free && it->size >= size) {
        // Found suitable block
        std::size_t remainder = it->size - size;

        if (remainder > min_alloc_size_) {
          // Split block
          void* alloc_ptr = it->ptr;
          void* remainder_ptr = static_cast<char*>(it->ptr) + size;

          // Update current block to allocated portion
          it->size = size;
          it->is_free = false;

          // Insert remainder as new free block
          auto next = std::next(it);
          blocks_.insert(next, block(remainder_ptr, remainder, true));
        } else {
          // Use entire block (remainder too small to split)
          it->is_free = false;
        }

        allocated_size_ += it->size;
        free_size_ -= it->size;

        return it->ptr;
      }
    }

    // No suitable block found, allocate new pool
    // Make sure new pool is large enough
    std::size_t new_pool_size = next_pool_size_;
    if (new_pool_size < size) {
      new_pool_size = size;
    }

    allocate_pool(new_pool_size);

    // Update next pool size for growth
    next_pool_size_ = static_cast<std::size_t>(next_pool_size_ * growth_factor_);

    // Retry allocation (should succeed now)
    return allocate(size);
  }

  //! @brief Deallocate memory
  //!
  //! Marks block as free and immediately coalesces with adjacent free blocks.
  //! Nullptr is a safe no-op.
  //!
  //! @param ptr Pointer to deallocate (nullptr is safe)
  //! @throws unknown_pointer_error if ptr is not from this pool
  //! @throws runtime_error if double-free detected
  void deallocate(void* ptr) override {
    if (!ptr) return;

    auto it = find_block(ptr);
    if (it == blocks_.end()) {
      UMPIRE_ERROR(unknown_pointer_error,
                   fmt::format("dynamic_pool_list: pointer {:p} not found in pool", ptr));
    }

    if (it->is_free) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("dynamic_pool_list: double free detected for pointer {:p}", ptr));
    }

    // Mark as free
    it->is_free = true;
    allocated_size_ -= it->size;
    free_size_ += it->size;

    // Coalesce with adjacent free blocks
    coalesce(it);
  }

  //! @brief Release free memory back to parent
  //!
  //! Returns completely free blocks to the parent. Only releases blocks
  //! that represent entire pool allocations (at pool boundaries).
  void release() {
    // Release free blocks that match pool boundaries
    // We need to check if a block corresponds to an entire pool
    auto block_it = blocks_.begin();
    while (block_it != blocks_.end()) {
      if (block_it->is_free) {
        // Check if this block matches a pool
        auto pool_it = std::find(pools_.begin(), pools_.end(), block_it->ptr);
        if (pool_it != pools_.end() && blocks_.size() > 1) {
          // This is an entire pool that's free, release it
          std::size_t block_size = block_it->size;
          parent_->deallocate(block_it->ptr);
          pools_.erase(pool_it);
          total_size_ -= block_size;
          free_size_ -= block_size;
          block_it = blocks_.erase(block_it);
        } else {
          ++block_it;
        }
      } else {
        ++block_it;
      }
    }
  }

  //! @brief Get total memory allocated from parent
  //! @return Total size in bytes
  std::size_t get_total_size() const { return total_size_; }

  //! @brief Get currently allocated memory (in use)
  //! @return Allocated size in bytes
  std::size_t get_allocated_size() const { return allocated_size_; }

  //! @brief Get currently free memory (available for allocation)
  //! @return Free size in bytes
  std::size_t get_free_size() const { return free_size_; }

  //! @brief Get initial pool size configuration
  //! @return Initial pool size in bytes
  std::size_t get_initial_pool_size() const { return initial_pool_size_; }

  //! @brief Get minimum allocation size for splitting
  //! @return Minimum allocation size in bytes
  std::size_t get_min_alloc_size() const { return min_alloc_size_; }

  //! @brief Get growth factor
  //! @return Growth factor multiplier
  double get_growth_factor() const { return growth_factor_; }

  //! @brief Get number of blocks in the list
  //! @return Number of blocks
  std::size_t get_block_count() const { return blocks_.size(); }
};

} // namespace strategy
} // namespace umpire

#endif // UMPIRE_strategy_dynamic_pool_list_HPP
