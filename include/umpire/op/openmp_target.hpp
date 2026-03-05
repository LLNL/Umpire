//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)

#include <omp.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>

#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"
#include "umpire/resource/platform.hpp"
#include "umpire/op/detail/utils.hpp"
#include "camp/resource.hpp"
#include "camp/resource/event.hpp"

namespace umpire {
namespace op {

// OpenMP Target implementation helpers
namespace {
// Helper function for device-to-device copy operations
template <typename T>
inline void copy_impl(T* src_ptr, T* dst_ptr, std::size_t count) {
  std::size_t size = detail::get_size<T>(count);
  
  #pragma omp target data use_device_ptr(src_ptr, dst_ptr)
  {
    std::memcpy(dst_ptr, src_ptr, size);
  }
}

// Helper function for host-to-device copy operations
template <typename T>
inline void host_to_device_copy_impl(T* src_ptr, T* dst_ptr, std::size_t count) {
  std::size_t size = detail::get_size<T>(count);
  
  #pragma omp target data use_device_ptr(dst_ptr)
  {
    std::memcpy(dst_ptr, src_ptr, size);
  }
}

// Helper function for device-to-host copy operations
template <typename T>
inline void device_to_host_copy_impl(T* src_ptr, T* dst_ptr, std::size_t count) {
  std::size_t size = detail::get_size<T>(count);
  
  #pragma omp target data use_device_ptr(src_ptr)
  {
    std::memcpy(dst_ptr, src_ptr, size);
  }
}

// IMPORTANT: OpenMP Target Async Limitations
// ===========================================
// The async implementations below are NOT truly asynchronous. OpenMP 5.x does not
// provide a mature mechanism for non-blocking device operations with event tracking
// compatible with CAMP's resource system.
//
// Current behavior:
//   - All "async" operations execute synchronously
//   - Return a completed event immediately after operation finishes
//   - No actual overlap with host computation
//
// This means:
//   - Performance: Same as synchronous operations
//   - Correctness: Safe, but no async performance benefit
//   - API: Maintains consistent interface with other platforms
//
// Future work: Consider using OpenMP tasks with dependencies for true async operations.

// Helper function for copy operations that returns an Event
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> copy_async_impl(
    T* src_ptr, T* dst_ptr, std::size_t count, camp::resources::Resource& res) {
  // Just call synchronous version and return a completed event
  copy_impl(src_ptr, dst_ptr, count);
  return camp::resources::EventProxy<camp::resources::Resource>{res};
}

// Helper function for host_to_device async copy
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> host_to_device_async_impl(
    T* src_ptr, T* dst_ptr, std::size_t count, camp::resources::Resource& res) {
  // Just call synchronous version and return a completed event
  host_to_device_copy_impl(src_ptr, dst_ptr, count);
  return camp::resources::EventProxy<camp::resources::Resource>{res};
}

// Helper function for device_to_host async copy
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> device_to_host_async_impl(
    T* src_ptr, T* dst_ptr, std::size_t count, camp::resources::Resource& res) {
  // Just call synchronous version and return a completed event
  device_to_host_copy_impl(src_ptr, dst_ptr, count);
  return camp::resources::EventProxy<camp::resources::Resource>{res};
}

// Helper function for memset operations
template <typename T>
inline void memset_impl(T* ptr, int val, std::size_t count) {
  std::size_t size = detail::get_size<T>(count);
  
  #pragma omp target data use_device_ptr(ptr)
  {
    std::memset(ptr, val, size);
  }
}

// Helper function for memset operations that returns an Event
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> memset_async_impl(
    T* ptr, int val, std::size_t count, camp::resources::Resource& res) {
  // Just call synchronous version and return a completed event
  memset_impl(ptr, val, count);
  return camp::resources::EventProxy<camp::resources::Resource>{res};
}
} // namespace

// Device-to-device copy specialization
template<>
struct copy<resource::omp_target_platform, resource::omp_target_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    copy_impl(src_ptr, dst_ptr, len);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(src_ptr, dst_ptr, len, res);
  }
};

// Host-to-device copy specialization
template<>
struct copy<resource::host_platform, resource::omp_target_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    host_to_device_copy_impl(src_ptr, dst_ptr, len);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return host_to_device_async_impl(src_ptr, dst_ptr, len, res);
  }
};

// Device-to-host copy specialization
template<>
struct copy<resource::omp_target_platform, resource::host_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    device_to_host_copy_impl(src_ptr, dst_ptr, len);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return device_to_host_async_impl(src_ptr, dst_ptr, len, res);
  }
};

// Memset specialization
template<>
struct memset<resource::omp_target_platform> {
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

// device_memset specialization
template<>
struct device_memset<resource::omp_target_platform> {
  template <typename T>
  static void exec(T* ptr, T val, std::size_t len) {
    if (!ptr || len == 0) {
      return;
    }

    // OpenMP target parallel loop
    #pragma omp target teams distribute parallel for is_device_ptr(ptr)
    for (std::size_t i = 0; i < len; i++) {
      ptr[i] = val;
    }
  }

  // Async version (OpenMP doesn't support true async, so call sync)
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* ptr, T val, std::size_t len, camp::resources::Resource& res) {
    exec(ptr, val, len);
    return camp::resources::EventProxy<camp::resources::Resource>{res};
  }
};

// Prefetch specialization
template<>
struct prefetch<resource::omp_target_platform> {
  template <typename T>
  static void exec(T* /*ptr*/, int /*device*/, std::size_t /*len*/) {
    // No-op: OpenMP Target doesn't provide explicit prefetch support
    // Data movement is handled automatically by the runtime
  }

  // Async version (also no-op)
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* /*ptr*/, int /*device*/, std::size_t /*len*/, camp::resources::Resource& res) {
    // No-op: OpenMP Target doesn't provide explicit prefetch support
    return camp::resources::EventProxy<camp::resources::Resource>{res};
  }
};

// Note: OpenMP Target platform uses the generic reallocate implementation from operations.hpp
// since direct OpenMP Target reallocation isn't supported and memory pools require a safe allocate-copy-free pattern

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_ENABLE_OPENMP_TARGET