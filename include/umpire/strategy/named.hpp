//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_named_HPP
#define UMPIRE_strategy_named_HPP

#include "umpire/strategy/allocation_strategy.hpp"

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
struct named_platform {
  using type = void;
};

template <typename Memory>
struct named_platform<Memory, std::void_t<typename Memory::platform>> {
  using type = typename Memory::platform;
};

} // namespace detail

//! @brief Lightweight naming wrapper for memory resources and strategies
//!
//! The named strategy gives a wrapped memory source a semantic allocator name
//! while delegating allocation behavior unchanged to the parent. Since the
//! memory base class stores the allocator name at construction time, passing the
//! custom name to allocation_strategy is sufficient to make the new allocator
//! visible through get_name() and the registry.
//!
//! @tparam Memory The memory source type to wrap (must inherit from memory)
template<typename Memory>
class named : public allocation_strategy {
public:
  //! @brief Platform type propagated from the wrapped memory source when available
  using platform = typename detail::named_platform<Memory>::type;

  //! @brief Construct a named wrapper around a memory source
  //!
  //! @param name Semantic name for this wrapper
  //! @param parent The memory source to wrap (must not be null)
  explicit named(const std::string& name, Memory* parent)
    : allocation_strategy(name, parent)
  {
  }

  //! @brief Delegate allocation directly to the parent
  //!
  //! @param size Number of bytes to allocate
  //! @return Pointer returned by the parent memory source
  void* allocate(std::size_t size) override
  {
    return parent_->allocate(size);
  }

  //! @brief Delegate deallocation directly to the parent
  //!
  //! @param ptr Pointer to deallocate
  void deallocate(void* ptr) override
  {
    parent_->deallocate(ptr);
  }
};

} // namespace strategy
} // namespace umpire

#endif // UMPIRE_strategy_named_HPP
