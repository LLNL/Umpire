#pragma once

#include "umpire/resource/platform.hpp"
#include "umpire/util/error.hpp"

#include <cstring>

namespace umpire {
namespace op {

namespace {
  // Generic implementation of host copy
  template<typename T>
  inline void copy_impl(T* src, T* dst, std::size_t len) {
    std::memcpy(dst, src, len * sizeof(T));
  }

  // Specialization for void*
  template<>
  inline void copy_impl<void>(void* src, void* dst, std::size_t len) {
    std::memcpy(dst, src, len);
  }

  // Generic implementation of host memset
  template<typename T>
  inline void memset_impl(T* src, int val, std::size_t len) {
    std::memset(src, val, sizeof(T) * len);
  }

  // Specialization for void*
  template<>
  inline void memset_impl<void>(void* src, int val, std::size_t len) {
    std::memset(src, val, len);
  }
}

// Host-to-host copy operation
template<>
struct copy<resource::host_platform, resource::host_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len);
  }
  
  // Async version returns a dummy event
  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    copy_impl(src, dst, len);
    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
  
  // Specialization for void*
  template<>
  static void exec<void>(void* src, void* dst, std::size_t len) {
    copy_impl<void>(src, dst, len);
  }
  
  template<>
  static camp::resources::Event exec<void>(void* src, void* dst, std::size_t len, camp::resources::Resource& r) {
    copy_impl<void>(src, dst, len);
    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
};

// Host memset operation
template<>
struct memset<resource::host_platform>
{
  template<typename T>
  static void exec(T* src, int val, std::size_t len) {
    memset_impl(src, val, len);
  }
  
  // Async version returns a dummy event
  template<typename T>
  static camp::resources::Event exec(T* src, int val, std::size_t len, camp::resources::Resource& r) {
    memset_impl(src, val, len);
    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
  
  // Specialization for void*
  template<>
  static void exec<void>(void* src, int val, std::size_t len) {
    memset_impl<void>(src, val, len);
  }
  
  template<>
  static camp::resources::Event exec<void>(void* src, int val, std::size_t len, camp::resources::Resource& r) {
    memset_impl<void>(src, val, len);
    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
};

// Host reallocate operation - uses system realloc
template<>
struct reallocate<resource::host_platform>
{
  template<typename T>
  static T* exec(T* src, std::size_t size) {
    if (!src) {
      // Return nullptr for nullptr input
      return nullptr;
    }
    
    if (size == 0) {
      if (src) {
        // Free memory for zero-sized allocation
        std::free(src);
      }
      return nullptr;
    }
    
    // Use standard realloc for host memory
    T* ret = static_cast<T*>(std::realloc(src, size * sizeof(T)));
    
    if (!ret && size > 0) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("Host realloc failed for pointer={}, size={}", 
                                    src, size * sizeof(T)));
    }
    
    return ret;
  }
  
  // Specialization for void* to handle size correctly
  template<>
  static void* exec<void>(void* src, std::size_t size) {
    if (!src) {
      return nullptr;
    }
    
    if (size == 0) {
      if (src) {
        std::free(src);
      }
      return nullptr;
    }
    
    void* ret = std::realloc(src, size);
    
    if (!ret && size > 0) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("Host realloc failed for pointer={}, size={}", 
                                    src, size));
    }
    
    return ret;
  }
};

}
}