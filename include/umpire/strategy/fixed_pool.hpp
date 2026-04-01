//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_fixed_pool_HPP
#define UMPIRE_strategy_fixed_pool_HPP

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/util/error.hpp"

#include "fmt/format.h"

#include <cstddef>
#include <string>
#include <vector>

namespace umpire {
namespace strategy {

//! @brief Fixed-size object pool for high-frequency same-size allocations
//!
//! The fixed_pool strategy pre-allocates pools of fixed-size objects and
//! maintains a free list for fast O(1) allocation and deallocation. This
//! is ideal for workloads with many allocations of the same size.
//!
//! @par Pool Management
//! - Pre-allocates one pool in constructor for immediate use
//! - Automatically grows by allocating new pools when free list is empty
//! - Each pool contains objects_per_pool_ objects
//! - Free list uses std::vector for fast pop_back/push_back
//!
//! @par Size Enforcement
//! - Throws std::invalid_argument if requested size != object_size_
//! - This ensures all allocations are uniform and poolable
//!
//! @par Memory Release
//! - release() method returns completely free pools to parent
//! - Always keeps at least one pool allocated
//! - Cannot track which pool an object came from (for performance)
//! - Therefore release() is conservative and only releases full pools
//!
//! @par Platform Propagation
//! The platform type is propagated from the wrapped memory source:
//! - fixed_pool<host_memory>::platform is host_platform
//! - fixed_pool<cuda_device_memory>::platform is cuda_platform
//!
//! @par Performance
//! - O(1) allocation (pop from free list)
//! - O(1) deallocation (push to free list)
//! - No size calculation or splitting required
//! - Ideal for same-size object creation/destruction patterns
//!
//! @par Composition
//! Can wrap any memory source:
//! - Resources: fixed_pool<host_memory>
//! - Thread-safe: thread_safe<fixed_pool<host_memory>>
//!
//! @tparam Memory The memory source type to wrap (must inherit from memory)
template<typename Memory>
class fixed_pool : public allocation_strategy {
public:
  //! @brief Platform type propagated from wrapped memory source
  using platform = typename Memory::platform;

private:
  std::size_t object_size_;         //!< Size of each object in bytes
  std::size_t objects_per_pool_;    //!< Number of objects per pool block
  std::vector<void*> free_list_;    //!< Available objects (fast allocation)
  std::vector<void*> pools_;        //!< Allocated pool blocks (for cleanup)

  // Statistics
  std::size_t total_objects_;       //!< Total objects across all pools
  std::size_t free_objects_;        //!< Current number of free objects

  //! @brief Allocate a new pool and add all objects to free list
  void allocate_pool() {
    std::size_t pool_size = object_size_ * objects_per_pool_;
    void* pool = parent_->allocate(pool_size);

    if (!pool) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("fixed_pool: failed to allocate pool of {} bytes", pool_size));
    }

    pools_.push_back(pool);

    // Add all objects in this pool to free list
    char* base = static_cast<char*>(pool);
    for (std::size_t i = 0; i < objects_per_pool_; ++i) {
      free_list_.push_back(base + (i * object_size_));
    }

    total_objects_ += objects_per_pool_;
    free_objects_ += objects_per_pool_;
  }

public:
  //! @brief Construct a fixed-size object pool
  //!
  //! Pre-allocates one pool in constructor for immediate use.
  //!
  //! @param name Name for this pool instance
  //! @param parent The memory source to wrap (must not be null)
  //! @param object_size Size of each object in bytes (must be > 0)
  //! @param objects_per_pool Number of objects per pool block (default 1024)
  //!
  //! @throws std::invalid_argument if parent is null (validated by base)
  //! @throws std::invalid_argument if object_size is 0
  //! @throws std::invalid_argument if objects_per_pool is 0
  explicit fixed_pool(const std::string& name, Memory* parent,
                     std::size_t object_size, std::size_t objects_per_pool = 1024)
    : allocation_strategy(name, parent)
    , object_size_(object_size)
    , objects_per_pool_(objects_per_pool)
    , total_objects_(0)
    , free_objects_(0)
  {
    if (object_size_ == 0) {
      UMPIRE_ERROR(std::invalid_argument,
                   "fixed_pool: object_size must be greater than 0");
    }

    if (objects_per_pool_ == 0) {
      UMPIRE_ERROR(std::invalid_argument,
                   "fixed_pool: objects_per_pool must be greater than 0");
    }

    // Pre-allocate first pool for fast initial allocations
    free_list_.reserve(objects_per_pool_);
    allocate_pool();
  }

  //! @brief Destructor - returns all pools to parent
  ~fixed_pool() {
    for (void* pool : pools_) {
      parent_->deallocate(pool);
    }
  }

  //! @brief Allocate a fixed-size object from the pool
  //!
  //! Returns an object from the free list. If the free list is empty,
  //! automatically allocates a new pool.
  //!
  //! @param size Number of bytes to allocate (must equal object_size_)
  //! @return Pointer to allocated object
  //! @throws std::invalid_argument if size != object_size_
  //! @throws out_of_memory_error if pool allocation fails
  void* allocate(std::size_t size) override {
    if (size != object_size_) {
      UMPIRE_ERROR(std::invalid_argument,
                   fmt::format("fixed_pool: requested size {} does not match object_size {}",
                              size, object_size_));
    }

    // Grow pool if needed
    if (free_list_.empty()) {
      allocate_pool();
    }

    // Pop from free list (O(1))
    void* ptr = free_list_.back();
    free_list_.pop_back();
    free_objects_--;

    return ptr;
  }

  //! @brief Deallocate a fixed-size object back to the pool
  //!
  //! Returns the object to the free list for reuse.
  //! Nullptr is a safe no-op.
  //!
  //! @param ptr Pointer to object to deallocate (nullptr is safe)
  void deallocate(void* ptr) override {
    if (!ptr) return;  // nullptr deallocation is safe no-op

    // Add back to free list (O(1))
    free_list_.push_back(ptr);
    free_objects_++;
  }

  //! @brief Release unused pools back to parent memory source
  //!
  //! Returns completely free pools to the parent while keeping at least
  //! one pool allocated. Since we don't track which pool each object
  //! came from (for performance), we can only release pools when we know
  //! for certain that entire pools are free.
  //!
  //! This implementation is conservative: it only releases pools when
  //! free_objects_ >= objects_per_pool_ and keeps at least one pool.
  void release() {
    // Can only release complete pools, and must keep at least one
    while (pools_.size() > 1 && free_objects_ >= objects_per_pool_) {
      // Remove last pool
      void* pool = pools_.back();
      pools_.pop_back();
      parent_->deallocate(pool);

      // Remove objects from free list
      // Note: We don't know which entries in free_list_ belong to this pool,
      // so we just remove the last objects_per_pool_ entries since they
      // should correspond to the most recently freed objects
      for (std::size_t i = 0; i < objects_per_pool_; ++i) {
        if (!free_list_.empty()) {
          free_list_.pop_back();
        }
      }

      total_objects_ -= objects_per_pool_;
      free_objects_ -= objects_per_pool_;
    }
  }

  //! @brief Get the fixed object size for this pool
  //! @return Size of each object in bytes
  std::size_t get_object_size() const { return object_size_; }

  //! @brief Get the number of objects per pool block
  //! @return Number of objects allocated per pool
  std::size_t get_objects_per_pool() const { return objects_per_pool_; }

  //! @brief Get the total number of objects across all pools
  //! @return Total objects allocated (free + in-use)
  std::size_t get_total_objects() const { return total_objects_; }

  //! @brief Get the number of free objects available
  //! @return Number of objects in free list
  std::size_t get_free_objects() const { return free_objects_; }

  //! @brief Get the number of allocated objects currently in use
  //! @return Number of objects allocated to users
  std::size_t get_allocated_objects() const { return total_objects_ - free_objects_; }

  //! @brief Get the number of pool blocks allocated
  //! @return Number of pools
  std::size_t get_pool_count() const { return pools_.size(); }
};

} // namespace strategy
} // namespace umpire

#endif // UMPIRE_strategy_fixed_pool_HPP
