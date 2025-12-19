#pragma once

#include <cstring>

#include "camp/resource.hpp"
#include "umpire/op/detail/utils.hpp"
#include "umpire/op/operations.hpp"
#include "umpire/resource/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

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
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* src, T* dst, std::size_t len,
                                                                     camp::resources::Resource& resource) noexcept
  {
    exec(src, dst, len);
    return detail::make_completed_event(resource);
  }
};

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
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* ptr, int val, std::size_t len,
                                                                     camp::resources::Resource& resource) noexcept
  {
    exec(ptr, val, len);
    return detail::make_completed_event(resource);
  }
};

template <>
struct prefetch<resource::host_platform> {
  /**
   * @brief Host memory prefetch (no-op)
   *
   * Prefetch is a no-op for host memory since the CPU has direct access.
   * This implementation exists for API compatibility and to avoid throwing errors.
   *
   * @tparam T Type of data being prefetched
   * @param ptr Pointer to memory (unused)
   * @param device Device ID (unused)
   * @param len Number of bytes to prefetch (unused)
   */
  template <typename T>
  static void exec(T* /*ptr*/, int /*device*/, std::size_t /*len*/) noexcept
  {
    // No-op: CPU already has direct access to host memory
  }

  /**
   * @brief Asynchronous host memory prefetch (no-op)
   *
   * Returns a completed event immediately since this is a no-op.
   */
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* /*ptr*/, int /*device*/, std::size_t /*len*/,
                                                                     camp::resources::Resource& resource) noexcept
  {
    // No-op: CPU already has direct access to host memory
    return detail::make_completed_event(resource);
  }
};

} // namespace op
} // namespace umpire
