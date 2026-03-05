//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_SYCL)

#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/sycl_compat.hpp"
#include "umpire/resource/platform.hpp"
#include "umpire/op/detail/utils.hpp"
#include "camp/resource.hpp"
#include "camp/resource/event.hpp"

#include <iostream>
#include <memory>
#include <sstream>

namespace umpire {
namespace op {

// SYCL helper functions
namespace detail {

/**
 * @brief Get SYCL queue from a resource
 *
 * @param resource The resource to get the queue from
 * @return sycl::queue& The SYCL queue
 */
inline sycl::queue& get_queue(camp::resources::Resource& resource)
{
  auto sycl_resource = resource.try_get<camp::resources::Sycl>();
  if (!sycl_resource) {
    UMPIRE_ERROR(resource_error,
                 fmt::format("Expected resources::Sycl, got resources::{}",
                            platform_to_string(resource.get_platform())));
  }
  return sycl_resource->get_queue();
}

} // namespace detail

// SYCL implementation helpers
namespace {
// Error handling for SYCL operations
inline void sycl_error_check(sycl::event event, const char* message) {
  try {
    event.wait_and_throw();
  } catch (const sycl::exception& e) {
    UMPIRE_ERROR(runtime_error, message + std::string(": ") + std::string(e.what()));
  }
}

// Synchronous copy implementation
template <typename T>
inline void copy_impl(T* src_ptr, T* dst_ptr, std::size_t count, sycl::usm::alloc alloc_type) {
  std::size_t size = detail::get_size<T>(count);
  
  sycl::queue queue;
  auto event = queue.memcpy(dst_ptr, src_ptr, size);
  sycl_error_check(event, "SYCL memcpy failed");
}

// Asynchronous copy implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> copy_async_impl(
    T* src_ptr, T* dst_ptr, std::size_t count, camp::resources::Resource& res,
    sycl::usm::alloc alloc_type) {

  std::size_t size = detail::get_size<T>(count);
  sycl::queue& queue = detail::get_queue(res);
  auto event = queue.memcpy(dst_ptr, src_ptr, size);

  return camp::resources::EventProxy<camp::resources::Resource>{res, event};
}

// Synchronous memset implementation
template <typename T>
inline void memset_impl(T* ptr, int val, std::size_t count) {
  std::size_t size = detail::get_size<T>(count);
  
  sycl::queue queue;
  auto event = queue.memset(ptr, val, size);
  sycl_error_check(event, "SYCL memset failed");
}

// Asynchronous memset implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> memset_async_impl(
    T* ptr, int val, std::size_t count, camp::resources::Resource& res) {

  std::size_t size = detail::get_size<T>(count);
  sycl::queue& queue = detail::get_queue(res);
  auto event = queue.memset(ptr, val, size);

  return camp::resources::EventProxy<camp::resources::Resource>{res, event};
}

// Synchronous prefetch implementation
template <typename T>
inline void prefetch_impl(T* ptr, int device, std::size_t count) {
  std::size_t size = detail::get_size<T>(count);
  
  sycl::queue queue;
  auto event = queue.prefetch(ptr, size);
  sycl_error_check(event, "SYCL prefetch failed");
}

// Asynchronous prefetch implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> prefetch_async_impl(
    T* ptr, int device, std::size_t count, camp::resources::Resource& res) {

  std::size_t size = detail::get_size<T>(count);
  sycl::queue& queue = detail::get_queue(res);
  auto event = queue.prefetch(ptr, size);

  return camp::resources::EventProxy<camp::resources::Resource>{res, event};
}
} // namespace

// Device-to-device copy specialization
template<>
struct copy<resource::sycl_platform, resource::sycl_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    copy_impl(src_ptr, dst_ptr, len, sycl::usm::alloc::device);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(src_ptr, dst_ptr, len, res, sycl::usm::alloc::device);
  }
};

// Host-to-device copy specialization
template<>
struct copy<resource::host_platform, resource::sycl_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    copy_impl(src_ptr, dst_ptr, len, sycl::usm::alloc::host);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(src_ptr, dst_ptr, len, res, sycl::usm::alloc::host);
  }
};

// Device-to-host copy specialization
template<>
struct copy<resource::sycl_platform, resource::host_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    copy_impl(src_ptr, dst_ptr, len, sycl::usm::alloc::host);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(src_ptr, dst_ptr, len, res, sycl::usm::alloc::host);
  }
};

// Memset specialization
template<>
struct memset<resource::sycl_platform> {
  template <typename T>
  static void exec(T* ptr, int val, std::size_t len) {
    memset_impl(ptr, val, len);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* ptr, int val, std::size_t len, camp::resources::Resource& res) {
    return memset_async_impl(ptr, val, len, res);
  }
};

// Prefetch specialization
template<>
struct prefetch<resource::sycl_platform> {
  template <typename T>
  static void exec(T* ptr, int device, std::size_t len) {
    prefetch_impl(ptr, device, len);
  }

  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* ptr, int device, std::size_t len, camp::resources::Resource& res) {
    return prefetch_async_impl(ptr, device, len, res);
  }
};

// Forward declaration for device_memset
namespace detail {
template <typename T>
void device_memset_sycl(T* ptr, T value, std::size_t count, sycl::queue& queue);
}

// device_memset specialization
template<>
struct device_memset<resource::sycl_platform> {
  template <typename T>
  static void exec(T* /*ptr*/, T /*val*/, std::size_t /*len*/) {
    // SYCL device_memset requires a queue context to execute properly.
    // Without a resource parameter, we cannot determine which device/queue to use.
    // Use the async version with a Resource parameter instead.
    UMPIRE_ERROR(runtime_error,
                 "device_memset for SYCL requires resource context. "
                 "Use the async version: device_memset(ptr, val, len, resource)");
  }

  // Async version with resource
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* ptr, T val, std::size_t len, camp::resources::Resource& res) {
    sycl::queue& queue = detail::get_queue(res);
    detail::device_memset_sycl(ptr, val, len, queue);
    return camp::resources::EventProxy<camp::resources::Resource>{res};
  }
};

// Note: SYCL platform uses the generic reallocate implementation from operations.hpp
// since direct SYCL reallocation isn't supported and memory pools require a safe allocate-copy-free pattern

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_ENABLE_SYCL