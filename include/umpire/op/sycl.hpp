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
#include "camp/resource.hpp"
#include "camp/resource/event.hpp"

#include <iostream>
#include <memory>
#include <sstream>

namespace umpire {
namespace op {

// Platform-specific type
struct sycl_platform {};

// SYCL implementation helpers
namespace {
// Size-aware calculation with type awareness
template<typename T>
inline std::size_t calculate_size(T* ptr, std::size_t count) {
  return std::is_same<T, void>::value ? count : count * sizeof(T);
}

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
  std::size_t size = calculate_size(src_ptr, count);
  
  sycl::queue queue;
  auto event = queue.memcpy(dst_ptr, src_ptr, size);
  sycl_error_check(event, "SYCL memcpy failed");
}

// Asynchronous copy implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> copy_async_impl(
    T* src_ptr, T* dst_ptr, std::size_t count, camp::resources::Resource& res,
    sycl::usm::alloc alloc_type) {
  
  std::size_t size = calculate_size(src_ptr, count);
  
  auto& sycl_res = dynamic_cast<camp::resources::Sycl&>(res);
  sycl::queue& queue = sycl_res.get_queue();
  
  auto event = queue.memcpy(dst_ptr, src_ptr, size);
  
  return camp::resources::EventProxy<camp::resources::Resource>{res, event};
}

// Synchronous memset implementation
template <typename T>
inline void memset_impl(T* ptr, int val, std::size_t count) {
  std::size_t size = calculate_size(ptr, count);
  
  sycl::queue queue;
  auto event = queue.memset(ptr, val, size);
  sycl_error_check(event, "SYCL memset failed");
}

// Asynchronous memset implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> memset_async_impl(
    T* ptr, int val, std::size_t count, camp::resources::Resource& res) {
  
  std::size_t size = calculate_size(ptr, count);
  
  auto& sycl_res = dynamic_cast<camp::resources::Sycl&>(res);
  sycl::queue& queue = sycl_res.get_queue();
  
  auto event = queue.memset(ptr, val, size);
  
  return camp::resources::EventProxy<camp::resources::Resource>{res, event};
}

// Synchronous prefetch implementation
template <typename T>
inline void prefetch_impl(T* ptr, int device, std::size_t count) {
  std::size_t size = calculate_size(ptr, count);
  
  sycl::queue queue;
  auto event = queue.prefetch(ptr, size);
  sycl_error_check(event, "SYCL prefetch failed");
}

// Asynchronous prefetch implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> prefetch_async_impl(
    T* ptr, int device, std::size_t count, camp::resources::Resource& res) {
  
  std::size_t size = calculate_size(ptr, count);
  
  auto& sycl_res = dynamic_cast<camp::resources::Sycl&>(res);
  sycl::queue& queue = sycl_res.get_queue();
  
  auto event = queue.prefetch(ptr, size);
  
  return camp::resources::EventProxy<camp::resources::Resource>{res, event};
}
} // namespace

// Device-to-device copy specialization
template<>
struct copy<sycl_platform, sycl_platform> {
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
struct copy<resource::host_platform, sycl_platform> {
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
struct copy<sycl_platform, resource::host_platform> {
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
struct memset<sycl_platform> {
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
struct prefetch<sycl_platform> {
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

// Reallocate operations - stub implementation
template<>
struct reallocate<sycl_platform> {
  template <typename T>
  static T* exec(T* src_ptr, std::size_t size) {
    // SYCL needs ResourceManager for allocation information
    UMPIRE_ERROR(runtime_error, "Direct SYCL reallocate not implemented");
    return nullptr;
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, std::size_t size, camp::resources::Resource& res) {
    UMPIRE_ERROR(runtime_error, "Direct SYCL async reallocate not implemented");
    return camp::resources::EventProxy<camp::resources::Resource>{res};
  }
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_ENABLE_SYCL