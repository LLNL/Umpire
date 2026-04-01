//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_thread_safe_HPP
#define UMPIRE_strategy_thread_safe_HPP

#include "umpire/strategy/allocation_strategy.hpp"

#include <mutex>
#include <string>

namespace umpire {
namespace strategy {

//! @brief Thread-safe wrapper strategy using mutex synchronization
//!
//! The thread_safe strategy wraps any memory source (resource or strategy)
//! and makes it safe for concurrent access from multiple threads by using
//! a std::mutex to serialize allocate() and deallocate() operations.
//!
//! @par Thread Safety
//! - All allocate() and deallocate() calls are serialized via std::mutex
//! - Exception-safe: mutex is released even if parent throws
//! - Uses RAII (std::lock_guard) for automatic lock management
//!
//! @par Platform Propagation
//! The platform type is propagated from the wrapped memory source:
//! - thread_safe<host_memory>::platform is host_platform
//! - thread_safe<cuda_device_memory>::platform is cuda_platform
//!
//! @par Performance
//! - Minimal critical section (just the allocate/deallocate call)
//! - Uses non-recursive std::mutex (lighter than recursive_mutex)
//! - Consider using with pooling strategies to reduce contention:
//!   thread_safe<fixed_pool<host_memory>> provides better throughput
//!   than thread_safe<host_memory> for high-frequency allocations
//!
//! @par Composition
//! Can wrap any memory source:
//! - Resources: thread_safe<host_memory>
//! - Strategies: thread_safe<fixed_pool<host_memory>>
//! - Other thread_safe: thread_safe<thread_safe<M>> (redundant but safe)
//!
//! @tparam Memory The memory source type to wrap (must inherit from memory)
template<typename Memory>
class thread_safe : public allocation_strategy {
public:
  //! @brief Platform type propagated from wrapped memory source
  using platform = typename Memory::platform;

private:
  mutable std::mutex mutex_;  //!< Mutex for serializing operations

public:
  //! @brief Construct a thread-safe wrapper around a memory source
  //!
  //! @param name Name for this thread-safe wrapper
  //! @param parent The memory source to wrap (must not be null)
  //!
  //! @throws std::invalid_argument if parent is null (validated by base)
  explicit thread_safe(const std::string& name, Memory* parent)
    : allocation_strategy(name, parent)
  {
  }

  //! @brief Thread-safe allocation
  //!
  //! Serializes access to the parent's allocate() method using mutex.
  //! The mutex is automatically released even if parent throws.
  //!
  //! @param size Number of bytes to allocate
  //! @return Pointer to allocated memory
  //! @throws Any exception thrown by parent's allocate()
  void* allocate(std::size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    return parent_->allocate(size);
  }

  //! @brief Thread-safe deallocation
  //!
  //! Serializes access to the parent's deallocate() method using mutex.
  //! The mutex is automatically released even if parent throws.
  //!
  //! @param ptr Pointer to memory to deallocate
  //! @throws Any exception thrown by parent's deallocate()
  void deallocate(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    parent_->deallocate(ptr);
  }
};

} // namespace strategy
} // namespace umpire

#endif // UMPIRE_strategy_thread_safe_HPP
