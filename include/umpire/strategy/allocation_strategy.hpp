//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_allocation_strategy_HPP
#define UMPIRE_allocation_strategy_HPP

#include "umpire/memory.hpp"

#include <stdexcept>
#include <string>

namespace umpire {

//! @brief Abstract base class for all allocation strategy decorators
//!
//! The allocation_strategy class implements the decorator pattern to wrap
//! other memory sources (resources or other strategies) and modify their
//! behavior. Strategies can add functionality like pooling, thread safety,
//! size limiting, etc.
//!
//! @par Decorator Pattern
//! Each strategy wraps a parent memory source via the parent_ pointer.
//! Derived strategies implement their own allocate/deallocate logic that
//! may delegate to the parent with modifications.
//!
//! @par Composition
//! Strategies can be composed by wrapping other strategies:
//! - Example: thread_safe<fixed_pool<host_memory>>
//!   - Innermost: host_memory (resource)
//!   - Middle: fixed_pool (strategy wrapping host_memory)
//!   - Outermost: thread_safe (strategy wrapping fixed_pool)
//!
//! @par Platform Transparency
//! Strategies delegate get_platform() to their parent, as they don't
//! change the underlying platform characteristics.
class allocation_strategy : public memory {
protected:
  memory* parent_;  //!< The wrapped memory source

public:
  //! @brief Construct a strategy that wraps a parent memory source
  //!
  //! @param name Name for this strategy instance
  //! @param parent The memory source to wrap (must not be null)
  //!
  //! @throws std::invalid_argument if parent is null
  explicit allocation_strategy(const std::string& name, memory* parent)
    : memory(name)
    , parent_(parent)
  {
    if (!parent_) {
      throw std::invalid_argument("allocation_strategy: parent cannot be null");
    }
  }

  //! @brief Virtual destructor
  ~allocation_strategy() override = default;

  //! @brief Get the platform of the underlying memory
  //!
  //! Delegates to parent since strategies don't change platform.
  //!
  //! @return The platform of the parent memory source
  camp::resources::Platform get_platform() const override {
    return parent_->get_platform();
  }

  //! @brief Access the parent memory source
  //!
  //! Provided for derived classes to access the wrapped memory source.
  //!
  //! @return Pointer to the parent memory source
  memory* get_parent() const { return parent_; }

  // Note: allocate() and deallocate() remain pure virtual from memory base
  // Each strategy implements its own allocation logic
};

} // namespace umpire

#endif // UMPIRE_allocation_strategy_HPP
