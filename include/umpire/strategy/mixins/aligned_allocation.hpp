//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_mixins_aligned_allocation_HPP
#define UMPIRE_strategy_mixins_aligned_allocation_HPP

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_map>

#include "umpire/memory.hpp"

namespace umpire {
namespace strategy {
namespace mixins {

//! @brief Aligned-allocation support for API v2 pool strategies.
//!
//! Provides alignment rounding plus over-allocate-and-offset aligned
//! allocation on top of any API v2 memory source. A pointer map recovers the
//! original base pointer on deallocation, so the wrapped memory source only
//! ever sees the pointers it produced.
//!
//! This is the API v2 port of `strategy::mixins::AlignedAllocation`.
class aligned_allocation {
public:
  aligned_allocation() = delete;

  //! @brief Configure alignment over a parent memory source.
  //!
  //! @param alignment Alignment in bytes; must be a power of two
  //! @param parent Memory source that provides the underlying storage
  aligned_allocation(std::size_t alignment, memory* parent)
    : aligned_parent_{parent}, alignment_{alignment}, mask_{static_cast<uintptr_t>(~(alignment - 1))}
  {
  }

  //! @brief Round `size` up to an integral multiple of the configured alignment.
  std::size_t aligned_round_up(std::size_t size) const
  {
    return size + (alignment_ - 1) - (size - 1) % alignment_;
  }

  //! @brief Allocate `size` bytes aligned on the configured boundary.
  void* aligned_allocate(std::size_t size)
  {
    std::size_t total_bytes{size + alignment_};
    uintptr_t ptr{reinterpret_cast<uintptr_t>(aligned_parent_->allocate(total_bytes))};

    uintptr_t alignment{alignment_ - 1};
    void* aligned_ptr{reinterpret_cast<void*>((ptr + alignment) & mask_)};

    base_pointer_map_[aligned_ptr] = std::make_tuple(reinterpret_cast<void*>(ptr), total_bytes);

    return aligned_ptr;
  }

  //! @brief Deallocate a pointer previously returned by aligned_allocate().
  void aligned_deallocate(void* ptr)
  {
    auto ptr_info = base_pointer_map_[ptr];
    void* buffer{std::get<0>(ptr_info)};
    base_pointer_map_.erase(ptr);
    aligned_parent_->deallocate(buffer);
  }

protected:
  //! @brief Parent memory source used for the underlying allocations.
  memory* aligned_parent_;

private:
  std::unordered_map<void*, std::tuple<void*, std::size_t>> base_pointer_map_;
  std::size_t alignment_;
  uintptr_t mask_;
};

} // namespace mixins
} // namespace strategy
} // namespace umpire

#endif // UMPIRE_strategy_mixins_aligned_allocation_HPP
