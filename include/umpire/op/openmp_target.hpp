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

#include "camp/resource.hpp"
#include "camp/resource/event.hpp"

namespace umpire {
namespace op {

struct openmp_target_platform {};

// Helper function for copy operations
template <typename T>
inline void copy_impl(T* src_ptr, T* dst_ptr, std::size_t len) {
  #pragma omp target data use_device_ptr(src_ptr, dst_ptr)
  {
    std::memcpy(dst_ptr, src_ptr, len * sizeof(T));
  }
}

// Helper function for copy operations that returns an Event
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> copy_async_impl(
    T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    
  #pragma omp target data use_device_ptr(src_ptr, dst_ptr)
  {
    std::memcpy(dst_ptr, src_ptr, len * sizeof(T));
  }
  
  // OpenMP Target doesn't have async operations, so we just return a completed event
  return camp::resources::EventProxy<camp::resources::Resource>{res};
}

// Helper function for memset operations
template <typename T>
inline void memset_impl(T* ptr, int val, std::size_t len) {
  #pragma omp target data use_device_ptr(ptr)
  {
    std::memset(ptr, val, len * sizeof(T));
  }
}

// Helper function for memset operations that returns an Event
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> memset_async_impl(
    T* ptr, int val, std::size_t len, camp::resources::Resource& res) {
    
  #pragma omp target data use_device_ptr(ptr)
  {
    std::memset(ptr, val, len * sizeof(T));
  }
  
  // OpenMP Target doesn't have async operations, so we just return a completed event
  return camp::resources::EventProxy<camp::resources::Resource>{res};
}

// Device-to-device copy specialization
template<>
struct copy<openmp_target_platform, openmp_target_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    copy_impl(src_ptr, dst_ptr, len);
  }
  
  // void pointer specialization
  static void exec(void* src_ptr, void* dst_ptr, std::size_t len) {
    copy_impl(static_cast<char*>(src_ptr), static_cast<char*>(dst_ptr), len);
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(src_ptr, dst_ptr, len, res);
  }
  
  // void pointer specialization
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      void* src_ptr, void* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    return copy_async_impl(static_cast<char*>(src_ptr), static_cast<char*>(dst_ptr), len, res);
  }
};

// Host-to-device copy specialization
template<>
struct copy<resource::host_platform, openmp_target_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    #pragma omp target data use_device_ptr(dst_ptr)
    {
      std::memcpy(dst_ptr, src_ptr, len * sizeof(T));
    }
  }
  
  // void pointer specialization
  static void exec(void* src_ptr, void* dst_ptr, std::size_t len) {
    #pragma omp target data use_device_ptr(dst_ptr)
    {
      std::memcpy(dst_ptr, src_ptr, len * sizeof(T));
    }
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    #pragma omp target data use_device_ptr(dst_ptr)
    {
      std::memcpy(dst_ptr, src_ptr, len * sizeof(T));
    }
    
    return camp::resources::EventProxy<camp::resources::Resource>{res};
  }
  
  // void pointer specialization
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      void* src_ptr, void* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    #pragma omp target data use_device_ptr(dst_ptr)
    {
      std::memcpy(dst_ptr, src_ptr, len * sizeof(char) * len);
    }
    
    return camp::resources::EventProxy<camp::resources::Resource>{res};
  }
};

// Device-to-host copy specialization
template<>
struct copy<openmp_target_platform, resource::host_platform> {
  template <typename T>
  static void exec(T* src_ptr, T* dst_ptr, std::size_t len) {
    #pragma omp target data use_device_ptr(src_ptr)
    {
      std::memcpy(dst_ptr, src_ptr, len * sizeof(T));
    }
  }
  
  // void pointer specialization
  static void exec(void* src_ptr, void* dst_ptr, std::size_t len) {
    #pragma omp target data use_device_ptr(src_ptr)
    {
      std::memcpy(dst_ptr, src_ptr, len * sizeof(char));
    }
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, T* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    #pragma omp target data use_device_ptr(src_ptr)
    {
      std::memcpy(dst_ptr, src_ptr, len * sizeof(T));
    }
    
    return camp::resources::EventProxy<camp::resources::Resource>{res};
  }
  
  // void pointer specialization
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      void* src_ptr, void* dst_ptr, std::size_t len, camp::resources::Resource& res) {
    #pragma omp target data use_device_ptr(src_ptr)
    {
      std::memcpy(dst_ptr, src_ptr, len * sizeof(char));
    }
    
    return camp::resources::EventProxy<camp::resources::Resource>{res};
  }
};

// Memset specialization
template<>
struct memset<openmp_target_platform> {
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

// Reallocate operations - basic implementation
template<>
struct reallocate<openmp_target_platform> {
  template <typename T>
  static T* exec(T* src_ptr, std::size_t size) {
    // For OpenMP Target, we need a strategy that involves:
    // 1. Allocate new memory
    // 2. Copy data if src_ptr is not null
    // 3. Free old memory if src_ptr is not null
    // 
    // This requires allocation information which is not available
    // in this layer, so it's implemented in ResourceManager
    UMPIRE_ERROR(runtime_error, "Direct OpenMP Target reallocate not implemented");
    return nullptr;
  }
  
  template <typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src_ptr, std::size_t size, camp::resources::Resource& res) {
    UMPIRE_ERROR(runtime_error, "Direct OpenMP Target async reallocate not implemented");
    return camp::resources::EventProxy<camp::resources::Resource>{res};
  }
};

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_ENABLE_OPENMP_TARGET