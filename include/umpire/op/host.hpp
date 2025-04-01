#pragma once

#include <cstring>

#include "umpire/op/detail/utils.hpp"
#include "umpire/op/operations.hpp"
#include "umpire/resource/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

// Host-to-host copy operation
template <>
struct copy<resource::host_platform, resource::host_platform> {
  /**
   * @brief Host-to-host memory copy implementation
   * 
   * @tparam T Type of data being copied
   * @param src Source pointer
   * @param dst Destination pointer
   * @param len Number of elements to copy
   */
  template <typename T>
  static void exec(T* src, T* dst, std::size_t len) noexcept
  {
    std::memcpy(dst, src, detail::get_size<T>(len));
  }

  /**
   * @brief Asynchronous host-to-host memory copy implementation
   * 
   * Since host operations are synchronous, this simply performs a sync copy
   * and returns a completed event.
   */
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src, T* dst, std::size_t len, camp::resources::Resource& resource) noexcept
  {
    exec(src, dst, len);
    return detail::make_completed_event(resource);
  }
};

// Host memset operation
template <>
struct memset<resource::host_platform> {
  /**
   * @brief Fill host memory with a value
   * 
   * @tparam T Type of data being set
   * @param ptr Pointer to memory
   * @param val Value to set (treated as byte)
   * @param len Number of elements to set
   */
  template <typename T>
  static void exec(T* ptr, int val, std::size_t len) noexcept
  {
    std::memset(ptr, val, detail::get_size<T>(len));
  }

  /**
   * @brief Asynchronous memset implementation for host memory
   * 
   * Since host operations are synchronous, this simply performs a sync memset
   * and returns a completed event.
   */
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* ptr, int val, std::size_t len, camp::resources::Resource& resource) noexcept
  {
    exec(ptr, val, len);
    return detail::make_completed_event(resource);
  }
};

// Host reallocate operation - uses system realloc
template <>
struct reallocate<resource::host_platform> {
  /**
   * @brief Reallocate host memory
   * 
   * @tparam T Type of data being reallocated
   * @param src Pointer to current allocation (may be null)
   * @param size New size in elements (or bytes for void*)
   * @return T* Pointer to new allocation or null on failure
   */
  template <typename T>
  static T* exec(T* src, std::size_t size)
  {
    // Special cases for null pointer or zero size
    if (!src)
      return nullptr;
    
    if (size == 0) {
      std::free(src);
      return nullptr;
    }

    // Calculate size in bytes based on type
    const std::size_t bytes = detail::get_size<T>(size);
    
    // Perform the reallocation
    T* ret = static_cast<T*>(std::realloc(src, bytes));

    // Error handling
    if (!ret && size > 0) {
      UMPIRE_ERROR(runtime_error, fmt::format("Host realloc failed for pointer={}, size={}", src, bytes));
    }

    return ret;
  }
};

// Host prefetch operation - no-op for host memory
template <>
struct prefetch<resource::host_platform> {
  /**
   * @brief Prefetch host memory (no-op)
   * 
   * This is a no-op for host memory as prefetching isn't applicable.
   */
  template <typename T>
  static void exec(T* UMPIRE_UNUSED_ARG(ptr), int UMPIRE_UNUSED_ARG(device), 
                   std::size_t UMPIRE_UNUSED_ARG(len)) noexcept
  {
    // No-op for host memory
  }

  /**
   * @brief Asynchronous prefetch for host memory (no-op)
   * 
   * This is a no-op that returns a completed event.
   */
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* UMPIRE_UNUSED_ARG(ptr), int UMPIRE_UNUSED_ARG(device), 
      std::size_t UMPIRE_UNUSED_ARG(len), camp::resources::Resource& resource) noexcept
  {
    // No-op for host memory
    return detail::make_completed_event(resource);
  }
};

} // namespace op
} // namespace umpire
