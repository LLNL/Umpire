//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_allocator_HPP
#define UMPIRE_allocator_HPP

#include "umpire/memory.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace umpire {

/*!
 * \brief STL-compatible typed allocator backed by an API v2 memory source.
 *
 * This adapter lets standard-library containers, `std::allocate_shared`, and
 * other allocator-aware facilities allocate through a concrete API v2 resource
 * or strategy while preserving compile-time platform information.
 *
 * \tparam T Value type allocated by the allocator.
 * \tparam Memory Concrete API v2 memory source type.
 */
template<typename T, typename Memory>
class allocator {
  static_assert(std::is_base_of_v<memory, Memory>,
                "allocator<T, Memory> requires Memory to derive from umpire::memory");

public:
  //! \brief STL-required value type.
  using value_type = T;
  //! \brief STL-required size type.
  using size_type = std::size_t;
  //! \brief STL-required difference type.
  using difference_type = std::ptrdiff_t;
  //! \brief Pointer type returned from allocation.
  using pointer = T*;
  //! \brief Const-qualified pointer type.
  using const_pointer = const T*;
  //! \brief Reference type for `T`.
  using reference = T&;
  //! \brief Const-qualified reference type for `T`.
  using const_reference = const T&;
  //! \brief Platform tag propagated from the wrapped memory source.
  using platform = typename Memory::platform;

  //! \brief Rebind this allocator to another value type.
  template<typename U>
  struct rebind {
    using other = allocator<U, Memory>;
  };

  /*!
   * \brief Construct an allocator bound to a concrete memory source.
   *
   * \param memory_source Memory resource or strategy to use for allocations.
   *
   * \throws std::invalid_argument if `memory_source` is null.
   */
  explicit allocator(Memory* memory_source)
    : memory_(memory_source)
  {
    if (!memory_) {
      throw std::invalid_argument("allocator: memory source cannot be null");
    }
  }

  allocator(const allocator&) = default;
  allocator& operator=(const allocator&) = default;

  /*!
   * \brief Convert from another `allocator<U, Memory>` sharing the same source.
   *
   * \tparam U Other value type.
   * \param other Allocator whose memory source should be reused.
   *
   * \note This conversion is intentionally implicit to satisfy allocator-aware
   *       container rebinding requirements.
   */
  template<typename U>
  // cppcheck-suppress noExplicitConstructor
  allocator(const allocator<U, Memory>& other)
    : memory_(other.get_memory())
  {
  }

  /*!
   * \brief Rebind assignment from another value-specialization.
   *
   * \tparam U Other value type.
   * \param other Allocator whose memory source should be reused.
   * \return `*this`.
   */
  template<typename U>
  allocator& operator=(const allocator<U, Memory>& other) noexcept
  {
    memory_ = other.get_memory();
    return *this;
  }

  /*!
   * \brief Allocate storage for `n` objects of type `T`.
   *
   * \param n Number of objects to allocate.
   * \return Pointer to uninitialized storage for `n` objects.
   *
   * \throws Any exception raised by the underlying memory source.
   */
  pointer allocate(size_type n)
  {
    return static_cast<pointer>(memory_->allocate(n * sizeof(T)));
  }

  /*!
   * \brief Deallocate storage previously obtained from this allocator.
   *
   * The element count parameter is accepted for STL compatibility and ignored
   * because API v2 deallocation is pointer-based.
   *
   * \param ptr Pointer to the storage to release.
   * \param size Unused element count supplied by the caller.
   */
  void deallocate(pointer ptr, size_type)
  {
    memory_->deallocate(static_cast<void*>(ptr));
  }

  //! \brief Return the maximum number of `T` objects representable by this allocator.
  // cppcheck-suppress functionStatic
  size_type max_size() const noexcept
  {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }

  //! \brief Access the bound memory source.
  Memory* get_memory() const noexcept
  {
    return memory_;
  }

  //! \brief Return the registry identifier of the bound memory source.
  int get_id() const noexcept
  {
    return memory_->get_id();
  }

  //! \brief Return the human-readable name of the bound memory source.
  const std::string& get_name() const noexcept
  {
    return memory_->get_name();
  }

  //! \brief Return the live allocation count expressed in `T` elements.
  size_type get_current_size() const noexcept
  {
    return memory_->get_current_size() / sizeof(T);
  }

  //! \brief Return the cumulative allocated size expressed in `T` elements.
  size_type get_actual_size() const noexcept
  {
    return memory_->get_actual_size() / sizeof(T);
  }

  //! \brief Return the high-water mark expressed in `T` elements.
  size_type get_highwatermark() const noexcept
  {
    return memory_->get_highwatermark() / sizeof(T);
  }

private:
  template<typename, typename>
  friend class allocator;

  Memory* memory_;
};

/*!
 * \brief Compare two typed allocators for source identity.
 *
 * \return `true` when both allocators point at the same memory source.
 */
template<typename T, typename U, typename Memory>
bool operator==(const allocator<T, Memory>& lhs, const allocator<U, Memory>& rhs) noexcept
{
  return lhs.get_memory() == rhs.get_memory();
}

//! \brief Negated form of `operator==` for typed allocators.
template<typename T, typename U, typename Memory>
bool operator!=(const allocator<T, Memory>& lhs, const allocator<U, Memory>& rhs) noexcept
{
  return !(lhs == rhs);
}

} // namespace umpire

#endif // UMPIRE_allocator_HPP
