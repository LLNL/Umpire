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

template<typename T, typename Memory>
class allocator {
  static_assert(std::is_base_of_v<memory, Memory>,
                "allocator<T, Memory> requires Memory to derive from umpire::memory");

public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using platform = typename Memory::platform;

  template<typename U>
  struct rebind {
    using other = allocator<U, Memory>;
  };

  explicit allocator(Memory* memory_source)
    : memory_(memory_source)
  {
    if (!memory_) {
      throw std::invalid_argument("allocator: memory source cannot be null");
    }
  }

  allocator(const allocator&) = default;
  allocator& operator=(const allocator&) = default;

  template<typename U>
  allocator(const allocator<U, Memory>& other)
    : memory_(other.get_memory())
  {
  }

  template<typename U>
  allocator& operator=(const allocator<U, Memory>& other) noexcept
  {
    memory_ = other.get_memory();
    return *this;
  }

  pointer allocate(size_type n)
  {
    return static_cast<pointer>(memory_->allocate(n * sizeof(T)));
  }

  void deallocate(pointer ptr, size_type)
  {
    memory_->deallocate(static_cast<void*>(ptr));
  }

  size_type max_size() const noexcept
  {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }

  Memory* get_memory() const noexcept
  {
    return memory_;
  }

  int get_id() const noexcept
  {
    return memory_->get_id();
  }

  const std::string& get_name() const noexcept
  {
    return memory_->get_name();
  }

  size_type get_current_size() const noexcept
  {
    return memory_->get_current_size() / sizeof(T);
  }

  size_type get_actual_size() const noexcept
  {
    return memory_->get_actual_size() / sizeof(T);
  }

  size_type get_highwatermark() const noexcept
  {
    return memory_->get_highwatermark() / sizeof(T);
  }

private:
  template<typename, typename>
  friend class allocator;

  Memory* memory_;
};

template<typename T, typename U, typename Memory>
bool operator==(const allocator<T, Memory>& lhs, const allocator<U, Memory>& rhs) noexcept
{
  return lhs.get_memory() == rhs.get_memory();
}

template<typename T, typename U, typename Memory>
bool operator!=(const allocator<T, Memory>& lhs, const allocator<U, Memory>& rhs) noexcept
{
  return !(lhs == rhs);
}

template<typename Memory>
using Allocator = allocator<char, Memory>;

} // namespace umpire

#endif // UMPIRE_allocator_HPP
