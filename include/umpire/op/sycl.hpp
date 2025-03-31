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

struct sycl_platform {};

// Error handling for SYCL operations
inline void sycl_error_check(sycl::event event, const char* message) {
  try {
    event.wait_and_throw();
  } catch (const sycl::exception& e) {
    UMPIRE_ERROR(runtime_error, message + std::string(": ") + std::string(e.what()));
  }
}

// Helper function for copy operations that returns an Event
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> copy_async_impl(
    T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res,
    sycl::usm::alloc alloc_type) {
  
  auto& sycl_res = dynamic_cast<camp::resources::Sycl&>(res);
  sycl::queue& queue = sycl_res.get_queue();
  
  auto event = queue.memcpy(dst_ptr, src_ptr, len * sizeof(T));
  
  return camp::resources::EventProxy<camp::resources::Resource>{res, event};
}

// Helper function for copy operations
template <typename T>
inline void copy_impl(T* src_ptr, T* dst_ptr, std::size_t len, sycl::usm::alloc alloc_type) {
  sycl::queue queue;
  auto event = queue.memcpy(dst_ptr, src_ptr, len * sizeof(T));
  sycl_error_check(event, "SYCL memcpy failed");
}

// Helper function for memset operations that returns an Event
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> memset_async_impl(
    T* ptr, int val, std::size_t len, camp::resources::Resource& res) {
  
  auto& sycl_res = dynamic_cast<camp::resources::Sycl&>(res);
  sycl::queue& queue = sycl_res.get_queue();
  
  auto event = queue.memset(ptr, val, len * sizeof(T));
  
  return camp::resources::EventProxy<camp::resources::Resource>{res, event};
}

// Helper function for memset operations
template <typename T>
inline void memset_impl(T* ptr, int val, std::size_t len) {
  sycl::queue queue;
  auto event = queue.memset(ptr, val, len * sizeof(T));
  sycl_error_check(event, "SYCL memset failed");
}

// Helper function for prefetch operations that returns an Event
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> prefetch_async_impl(
    T* ptr, int device, std::size_t len, camp::resources::Resource& res) {
  
  auto& sycl_res = dynamic_cast<camp::resources::Sycl&>(res);
  sycl::queue& queue = sycl_res.get_queue();
  
  auto event = queue.prefetch(ptr, len * sizeof(T));
  
  return camp::resources::EventProxy<camp::resources::Resource>{res, event};
}

// Helper function for prefetch operations
template <typename T>
inline void prefetch_impl(T* ptr, int device, std::size_t len) {
  sycl::queue queue;
  auto event = queue.prefetch(ptr, len * sizeof(T));
  sycl_error_check(event, "SYCL prefetch failed");
}

// Device-to-device copy specialization
template<>
struct copy<sycl_platform, sycl_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    copy_impl(src_ptr, dst_ptr, len, sycl::usm::alloc::device);
  }
  
  // void pointer specialization
  static void exec(void* src_ptr, void* dst_ptr, std::size_t len) {
    copy_impl(static_cast<char*>(src_ptr), static_cast<char*>(dst_ptr), len, sycl::usm::alloc::device);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(src_ptr, dst_ptr, len, res, sycl::usm::alloc::device);
  }
  
  // void pointer specialization
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      void* src_ptr, void* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(static_cast<char*>(src_ptr), static_cast<char*>(dst_ptr), len, res, sycl::usm::alloc::device);
  }
};

// Host-to-device copy specialization
template<>
struct copy<resource::host_platform, sycl_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    copy_impl(src_ptr, dst_ptr, len, sycl::usm::alloc::host);
  }
  
  // void pointer specialization
  static void exec(void* src_ptr, void* dst_ptr, std::size_t len) {
    copy_impl(static_cast<char*>(src_ptr), static_cast<char*>(dst_ptr), len, sycl::usm::alloc::host);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(src_ptr, dst_ptr, len, res, sycl::usm::alloc::host);
  }
  
  // void pointer specialization
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      void* src_ptr, void* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(static_cast<char*>(src_ptr), static_cast<char*>(dst_ptr), len, res, sycl::usm::alloc::host);
  }
};

// Device-to-host copy specialization
template<>
struct copy<sycl_platform, resource::host_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    copy_impl(src_ptr, dst_ptr, len, sycl::usm::alloc::host);
  }
  
  // void pointer specialization
  static void exec(void* src_ptr, void* dst_ptr, std::size_t len) {
    copy_impl(static_cast<char*>(src_ptr), static_cast<char*>(dst_ptr), len, sycl::usm::alloc::host);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(src_ptr, dst_ptr, len, res, sycl::usm::alloc::host);
  }
  
  // void pointer specialization
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      void* src_ptr, void* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(static_cast<char*>(src_ptr), static_cast<char*>(dst_ptr), len, res, sycl::usm::alloc::host);
  }
};

// Memset specialization
template<>
struct memset<sycl_platform> {
  template <typename T>
  static void exec(T* ptr, int val, std::size_t len) {
    memset_impl(ptr, val, len);
  }
  
  // void pointer specialization
  static void exec(void* ptr, int val, std::size_t len) {
    memset_impl(static_cast<char*>(ptr), val, len);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* ptr, int val, std::size_t len, camp::resources::Resource& res) {
    return memset_async_impl(ptr, val, len, res);
  }
  
  // void pointer specialization
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      void* ptr, int val, std::size_t len, camp::resources::Resource& res) {
    return memset_async_impl(static_cast<char*>(ptr), val, len, res);
  }
};

// Prefetch specialization
template<>
struct prefetch<sycl_platform> {
  template <typename T>
  static void exec(T* ptr, int device, std::size_t len) {
    prefetch_impl(ptr, device, len);
  }
  
  // void pointer specialization
  static void exec(void* ptr, int device, std::size_t len) {
    prefetch_impl(static_cast<char*>(ptr), device, len);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* ptr, int device, std::size_t len, camp::resources::Resource& res) {
    return prefetch_async_impl(ptr, device, len, res);
  }
  
  // void pointer specialization
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      void* ptr, int device, std::size_t len, camp::resources::Resource& res) {
    return prefetch_async_impl(static_cast<char*>(ptr), device, len, res);
  }
};

// Reallocate operations - basic implementation
template<>
struct reallocate<sycl_platform> {
  template <typename T>
  static T* exec(T* src_ptr, std::size_t size) {
    // For SYCL, we need a strategy that involves:
    // 1. Allocate new memory
    // 2. Copy data if src_ptr is not null
    // 3. Free old memory if src_ptr is not null
    // 
    // This requires allocation information which is not available
    // in this layer, so it's implemented in ResourceManager
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