//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_monotonic_buffer_HPP
#define UMPIRE_strategy_monotonic_buffer_HPP

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/util/error.hpp"

#include "fmt/format.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
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
struct monotonic_buffer_platform {
  using type = void;
};

template <typename Memory>
struct monotonic_buffer_platform<Memory, std::void_t<typename Memory::platform>> {
  using type = typename Memory::platform;
};

} // namespace detail

//! @brief Bump-pointer allocator with bulk reset semantics
//!
//! The monotonic_buffer strategy allocates a single backing block from its
//! parent and fulfills allocations by monotonically bumping an offset into that
//! block. Individual deallocations are ignored; call release() to reset the
//! buffer for reuse.
template<typename Memory>
class monotonic_buffer : public allocation_strategy {
public:
  //! @brief Platform type propagated from wrapped memory source when available
  using platform = typename detail::monotonic_buffer_platform<Memory>::type;

private:
  static constexpr std::size_t ALIGNMENT = alignof(std::max_align_t);

  void* buffer_;
  std::size_t capacity_;
  std::size_t current_offset_;
  std::size_t high_watermark_;

  static std::size_t align_up(std::size_t offset)
  {
    const std::size_t remainder = offset % ALIGNMENT;
    return remainder == 0 ? offset : offset + (ALIGNMENT - remainder);
  }

public:
  //! @brief Construct a monotonic buffer with a fixed backing block
  //!
  //! @param name Name for this buffer instance
  //! @param parent The memory source to wrap (must not be null)
  //! @param capacity Size of the backing block in bytes
  explicit monotonic_buffer(const std::string& name, Memory* parent, std::size_t capacity)
    : allocation_strategy(name, parent)
    , buffer_(nullptr)
    , capacity_(capacity)
    , current_offset_(0)
    , high_watermark_(0)
  {
    if (capacity_ == 0) {
      throw std::invalid_argument("monotonic_buffer: capacity must be greater than 0");
    }

    buffer_ = parent_->allocate(capacity_);
    if (!buffer_) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("monotonic_buffer: failed to allocate {} bytes", capacity_));
    }
  }

  //! @brief Destructor - returns the backing block to parent
  ~monotonic_buffer()
  {
    if (buffer_) {
      parent_->deallocate(buffer_);
    }
  }

  //! @brief Allocate memory by bumping the current offset
  //!
  //! @param size Number of bytes to allocate
  //! @return Pointer into the backing buffer, or nullptr for zero-size requests
  void* allocate(std::size_t size) override
  {
    if (size == 0) {
      return nullptr;
    }

    const std::size_t aligned_offset = align_up(current_offset_);
    if (aligned_offset > capacity_ || size > capacity_ - aligned_offset) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("monotonic_buffer: capacity exceeded for request {} (used {} / {})",
                               size, current_offset_, capacity_));
    }

    void* ptr = static_cast<char*>(buffer_) + aligned_offset;
    current_offset_ = aligned_offset + size;
    if (current_offset_ > high_watermark_) {
      high_watermark_ = current_offset_;
    }

    return ptr;
  }

  //! @brief Individual deallocation is intentionally unsupported
  //!
  //! @param ptr Pointer to ignore
  void deallocate(void* /* ptr */) override
  {
    // No-op: memory is reclaimed only via release() or destruction.
  }

  //! @brief Reset the buffer to the beginning for reuse
  void release()
  {
    current_offset_ = 0;
  }

  //! @brief Get the total backing capacity in bytes
  std::size_t get_capacity() const { return capacity_; }

  //! @brief Get bytes consumed since the last release
  std::size_t get_current_size() const { return current_offset_; }

  //! @brief Get the highest observed byte usage
  std::size_t get_high_watermark() const { return high_watermark_; }

  //! @brief Get remaining space before the next allocation would fail
  std::size_t get_remaining_capacity() const
  {
    const std::size_t aligned_offset = align_up(current_offset_);
    return aligned_offset >= capacity_ ? 0 : capacity_ - aligned_offset;
  }

  //! @brief Access the raw backing buffer acquired from the parent
  //!
  //! Exposed so callers that need to manage bump-pointer offsets themselves
  //! (e.g. a v1-compatibility bridge that must preserve an unaligned,
  //! zero-overhead bump-allocation contract different from this class's own
  //! `allocate()`) can still route buffer acquisition/release through this
  //! class's constructor/destructor.
  //!
  //! @return Pointer to the start of the backing buffer
  void* get_buffer() const { return buffer_; }
};

} // namespace strategy
} // namespace umpire

#endif // UMPIRE_strategy_monotonic_buffer_HPP
