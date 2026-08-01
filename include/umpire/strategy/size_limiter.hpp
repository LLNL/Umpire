//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_size_limiter_HPP
#define UMPIRE_strategy_size_limiter_HPP

#include "umpire/detail/registry.hpp"
#include "umpire/error.hpp"
#include "umpire/strategy/allocation_strategy.hpp"

#include "fmt/format.h"

#include <atomic>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

namespace umpire {
namespace strategy {

namespace detail {

// SFINAE helper mirroring strategy::detail::thread_safe_platform (see
// thread_safe.hpp): tolerates a `Memory` type without a `platform` member
// alias (e.g. a runtime-typed bridge such as
// strategy::detail::v1_backed_memory), defaulting to `void` instead of a
// hard compile error.
template <typename Memory, typename = void>
struct size_limiter_platform {
  using type = void;
};

template <typename Memory>
struct size_limiter_platform<Memory, std::void_t<typename Memory::platform>> {
  using type = typename Memory::platform;
};

} // namespace detail

//! @brief Decorator that caps the total live bytes allocated through a parent
//!
//! The size_limiter strategy wraps another memory source and rejects
//! allocations that would cause total live allocations to exceed a configured
//! byte limit.
//!
//! @par Limit Enforcement
//! - Allocation reserves bytes before delegating to parent
//! - If reservation would exceed limit, allocate() throws without calling parent
//! - If parent allocation throws, reservation is rolled back before rethrow
//!
//! @par Deallocation
//! - Uses the shared v2 allocation registry to recover the allocation size
//! - Decrements current usage only after parent deallocation succeeds
//! - nullptr deallocation is a safe no-op
//!
//! @par Platform Propagation
//! The platform type is propagated from the wrapped memory source.
//!
//! @tparam Memory The memory source type to wrap (must inherit from memory)
template<typename Memory>
class size_limiter : public allocation_strategy {
public:
  //! @brief Platform type propagated from the wrapped memory source when available
  using platform = typename detail::size_limiter_platform<Memory>::type;

private:
  std::size_t limit_;
  std::atomic<std::size_t> current_{0};

public:
  //! @brief Construct a size-limited strategy
  //!
  //! @param name Name for this limiter instance
  //! @param parent The memory source to wrap (must not be null)
  //! @param limit Maximum number of live bytes allowed through this strategy
  explicit size_limiter(const std::string& name, Memory* parent, std::size_t limit)
    : allocation_strategy(name, parent)
    , limit_(limit)
  {
  }

  //! @brief Allocate memory if doing so would not exceed the configured limit
  //!
  //! @param size Number of bytes to allocate
  //! @return Pointer returned by the parent memory source
  //! @throws umpire::logic_error if the allocation would exceed the limit
  //! @throws Any exception thrown by the parent allocate(), after rolling back
  //!         the reserved usage accounting
  void* allocate(std::size_t size) override
  {
    if (size == 0) {
      return parent_->allocate(size);
    }

    std::size_t observed = current_.load(std::memory_order_relaxed);
    while (true) {
      if (observed > limit_ || size > limit_ - observed) {
        throw umpire::logic_error(
          fmt::format("size_limiter: allocation of {} bytes exceeds limit {} with {} bytes in use",
                      size, limit_, observed));
      }

      if (current_.compare_exchange_weak(observed,
                                         observed + size,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
        break;
      }
    }

    try {
      return parent_->allocate(size);
    } catch (...) {
      current_.fetch_sub(size, std::memory_order_relaxed);
      throw;
    }
  }

  //! @brief Deallocate memory and return its size to the available quota
  //!
  //! @param ptr Pointer to memory to deallocate (nullptr is safe)
  //! @throws umpire::unknown_allocation if ptr is not tracked in the registry
  //! @throws Any exception thrown by the parent deallocate()
  void deallocate(void* ptr) override
  {
    if (!ptr) {
      return;
    }

    auto record = umpire::detail::registry::get().find_allocation(ptr);
    if (!record) {
      throw umpire::unknown_allocation(
        fmt::format("size_limiter: cannot determine allocation size for pointer {:p}", ptr));
    }

    const std::size_t size = record->size;
    parent_->deallocate(ptr);
    current_.fetch_sub(size, std::memory_order_relaxed);
  }

  //! @brief Get the configured byte limit
  std::size_t get_limit() const { return limit_; }

  //! @brief Get the number of currently live bytes counted by the limiter
  std::size_t get_current_usage() const
  {
    return current_.load(std::memory_order_relaxed);
  }
};

} // namespace strategy
} // namespace umpire

#endif // UMPIRE_strategy_size_limiter_HPP
