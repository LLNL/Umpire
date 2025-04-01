#pragma once

#include <cstring>

#include "umpire/op/detail/utils.hpp"
#include "umpire/resource/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

namespace {

template <typename T>
inline void copy_impl(T* src, T* dst, std::size_t len)
{
  std::memcpy(dst, src, detail::get_size<T>(len));
}

template <typename T>
inline void memset_impl(T* src, int val, std::size_t len)
{
  std::memset(src, val, detail::get_size<T>(len));
}

} // namespace

// Host-to-host copy operation
template <>
struct copy<resource::host_platform, resource::host_platform> {
  template <typename T>
  static void exec(T* src, T* dst, std::size_t len)
  {
    copy_impl(src, dst, len);
  }

  // Async version returns a dummy event
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* src, T* dst, std::size_t len,
                                                                     camp::resources::Resource& r)
  {
    copy_impl(src, dst, len);
    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
};

// Host memset operation
template <>
struct memset<resource::host_platform> {
  template <typename T>
  static void exec(T* src, int val, std::size_t len)
  {
    memset_impl(src, val, len);
  }

  // Async version returns a dummy event
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* src, int val, std::size_t len,
                                                                     camp::resources::Resource& r)
  {
    memset_impl(src, val, len);
    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
};

// Host reallocate operation - uses system realloc
template <>
struct reallocate<resource::host_platform> {
  template <typename T>
  static T* exec(T* src, std::size_t size)
  {
    if (!src)
      return nullptr;
    if (size == 0) {
      std::free(src);
      return nullptr;
    }

    // Calculate appropriate size
    const std::size_t bytes = std::is_same<T, void>::value ? size : size * sizeof(T);
    T* ret = static_cast<T*>(std::realloc(src, bytes));

    if (!ret && size > 0) {
      UMPIRE_ERROR(runtime_error, fmt::format("Host realloc failed for pointer={}, size={}", src, bytes));
    }

    return ret;
  }
};

// Host prefetch operation - no-op for host memory
template <>
struct prefetch<resource::host_platform> {
  template <typename T>
  static void exec(T* src, int device, std::size_t len)
  {
    // No-op for host memory
  }

  // Async version returns a dummy event
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* src, int device, std::size_t len,
                                                                     camp::resources::Resource& r)
  {
    // No-op for host memory
    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
};

} // namespace op
} // namespace umpire
