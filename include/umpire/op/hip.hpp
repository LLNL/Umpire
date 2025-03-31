#pragma once

#include "umpire/resource/platform.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/Platform.hpp"

// Forward declaration of kernel for launching directly in device code if needed
extern "C" {
__global__ void
umpire_hip_fill(void* data, int value, std::size_t length);
}

namespace {
  template<typename SRC, typename DST>
  struct get_kind;

  template<>
  struct get_kind<resource::hip_platform, resource::host_platform> {
    static constexpr hipMemcpyKind value = hipMemcpyDeviceToHost;
  };

  template<>
  struct get_kind<resource::host_platform, resource::hip_platform> {
    static constexpr hipMemcpyKind value = hipMemcpyHostToDevice;
  };

  template<>
  struct get_kind<resource::hip_platform, resource::hip_platform> {
    static constexpr hipMemcpyKind value = hipMemcpyDeviceToDevice;
  };
}

namespace umpire {
namespace op {

namespace {
  // Helper function to check if a HIP device supports managed memory features
  inline bool check_device_managed_memory(int device) {
    hipDeviceProp_t properties;
    hipError_t error = ::hipGetDeviceProperties(&properties, device);
    
    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("hipGetDeviceProperties for device {} failed with error: {}",
                                   device, hipGetErrorString(error)));
    }
    
    return (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1);
  }
  
  // Generic function to handle hipMemAdvise operations
  template<typename T>
  inline void advise_impl(T* ptr, std::size_t n, int device, hipMemoryAdvise advice) {
    std::size_t size = n;
    if (std::is_same<T, void>::value) {
      // void pointers don't have a size
    } else {
      size = sizeof(T) * n;
    }
    
    if (check_device_managed_memory(device)) {
      hipError_t error = ::hipMemAdvise(ptr, size, advice, device);

      if (error != hipSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("hipMemAdvise(ptr={}, size={}, advice={}, device={}) failed with error: {}",
                                     ptr, size, static_cast<int>(advice), device, hipGetErrorString(error)));
      }
    }
  }

  // Generic copy function that handles the different kinds of copies
  template<typename T>
  inline void copy_impl(T* src, T* dst, std::size_t len, hipMemcpyKind kind) {
    std::size_t size = len;
    if (!std::is_same<T, void>::value) {
      size = sizeof(T) * len;
    }
    
    hipError_t error = ::hipMemcpy(dst, src, size, kind);
    if (error != hipSuccess) {
      UMPIRE_ERROR(
          runtime_error,
          umpire::fmt::format(
              "hipMemcpy(dst={}, src={}, size={}, kind={}) failed with error: {}",
              dst, src, size, static_cast<int>(kind), hipGetErrorString(error)));
    }
  }

  // Async version of copy for use with HIP streams
  template<typename T>
  inline camp::resources::Event copy_async_impl(T* src, T* dst, std::size_t len, camp::resources::Resource& r, hipMemcpyKind kind) {
    auto device = r.try_get<camp::resources::Hip>();
    if (!device) {
      UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Hip, got resources::{}",
                                                    platform_to_string(r.get_platform())));
    }
    auto stream = device->get_stream();

    std::size_t size = len;
    if (!std::is_same<T, void>::value) {
      size = sizeof(T) * len;
    }

    hipError_t error = ::hipMemcpyAsync(dst, src, size, kind, stream);

    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("hipMemcpyAsync(dst={}, src={}, size={}, kind={}, stream={}) failed with error: {}",
                                    dst, src, size, static_cast<int>(kind), 
                                    (void*)stream, hipGetErrorString(error)));
    }

    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
  
  // Generic memset implementation
  template<typename T>
  inline void memset_impl(T* ptr, int value, std::size_t len) {
    std::size_t size = len;
    if (!std::is_same<T, void>::value) {
      size = sizeof(T) * len;
    }
    
    hipError_t error = ::hipMemset(ptr, value, size);

    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("hipMemset(ptr={}, value={}, size={}) failed with error: {}", 
                                    ptr, value, size, hipGetErrorString(error)));
    }
  }
  
  // Async version of memset
  template<typename T>
  inline camp::resources::Event memset_async_impl(T* ptr, int value, std::size_t len, camp::resources::Resource& r) {
    auto device = r.try_get<camp::resources::Hip>();
    if (!device) {
      UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Hip, got resources::{}",
                                                    platform_to_string(r.get_platform())));
    }
    auto stream = device->get_stream();

    std::size_t size = len;
    if (!std::is_same<T, void>::value) {
      size = sizeof(T) * len;
    }

    hipError_t error = ::hipMemsetAsync(ptr, value, size, stream);

    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format(
                    "hipMemsetAsync(ptr={}, value={}, size={}, stream={}) failed with error: {}",
                    ptr, value, size, (void*)stream, hipGetErrorString(error)));
    }

    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
}

// Copy operations for different platform combinations
template<>
struct copy<resource::hip_platform, resource::hip_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len, hipMemcpyDeviceToDevice);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    return copy_async_impl(src, dst, len, r, hipMemcpyDeviceToDevice);
  }
};

template<>
struct copy<resource::hip_platform, resource::host_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len, hipMemcpyDeviceToHost);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    return copy_async_impl(src, dst, len, r, hipMemcpyDeviceToHost);
  }
};

template<>
struct copy<resource::host_platform, resource::hip_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len, hipMemcpyHostToDevice);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    return copy_async_impl(src, dst, len, r, hipMemcpyHostToDevice);
  }
};

// Special handling for void pointers
template<>
template<>
inline void copy<resource::hip_platform, resource::hip_platform>::exec<void>(void* src, void* dst, std::size_t len) {
  copy_impl<void>(src, dst, len, hipMemcpyDeviceToDevice);
}

template<>
template<>
inline void copy<resource::hip_platform, resource::host_platform>::exec<void>(void* src, void* dst, std::size_t len) {
  copy_impl<void>(src, dst, len, hipMemcpyDeviceToHost);
}

template<>
template<>
inline void copy<resource::host_platform, resource::hip_platform>::exec<void>(void* src, void* dst, std::size_t len) {
  copy_impl<void>(src, dst, len, hipMemcpyHostToDevice);
}

// Memset operations
template<>
struct memset<resource::hip_platform>
{
  template<typename T>
  static void exec(T* src, int val, std::size_t len) {
    memset_impl(src, val, len);
  }

  template<typename T>
  static camp::resources::Event exec(T* src, int val, std::size_t len, camp::resources::Resource& r) {
    return memset_async_impl(src, val, len, r);
  }
  
  // Specialization for void*
  template<>
  static void exec<void>(void* src, int val, std::size_t len) {
    memset_impl<void>(src, val, len);
  }
  
  template<>
  static camp::resources::Event exec<void>(void* src, int val, std::size_t len, camp::resources::Resource& r) {
    return memset_async_impl<void>(src, val, len, r);
  }
};

// Reallocate implementation
template<>
struct reallocate<resource::hip_platform>
{
  template<typename T>
  static T* exec(T* src, std::size_t size) {
    if (!src) {
      // This should allocate memory, but we can't do that directly here
      // since we don't have access to the allocator
      return nullptr;
    }
    
    if (size == 0) {
      // Should deallocate src and return nullptr
      return nullptr;
    }
    
    // This should be handled by the ResourceManager which has access to:
    // 1. The AllocationRecord to get the original size
    // 2. The Allocator to allocate new memory
    
    // For now, just return nullptr to indicate this needs to be
    // handled at a higher level
    return nullptr;
  }
};

// Memory advice operations
template<>
struct accessed_by<resource::hip_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, hipMemAdviseSetAccessedBy);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, hipMemAdviseSetAccessedBy);
  }
};

template<>
struct preferred_location<resource::hip_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, hipMemAdviseSetPreferredLocation);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, hipMemAdviseSetPreferredLocation);
  }
};

template<>
struct read_mostly<resource::hip_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, hipMemAdviseSetReadMostly);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, hipMemAdviseSetReadMostly);
  }
};

template<>
struct unset_accessed_by<resource::hip_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, hipMemAdviseUnsetAccessedBy);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, hipMemAdviseUnsetAccessedBy);
  }
};

template<>
struct unset_preferred_location<resource::hip_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, hipMemAdviseUnsetPreferredLocation);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, hipMemAdviseUnsetPreferredLocation);
  }
};

template<>
struct unset_read_mostly<resource::hip_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, hipMemAdviseUnsetReadMostly);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, hipMemAdviseUnsetReadMostly);
  }
};

#if HIP_VERSION_MAJOR >= 5
template<>
struct coarse_grain<resource::hip_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, hipMemAdviseSetCoarseGrain);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, hipMemAdviseSetCoarseGrain);
  }
};

template<>
struct unset_coarse_grain<resource::hip_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, hipMemAdviseUnsetCoarseGrain);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, hipMemAdviseUnsetCoarseGrain);
  }
};
#endif

// Prefetch operations
template<>
struct prefetch<resource::hip_platform>
{
  template<typename T>
  static void exec(T* src, int device, std::size_t len) {
    // Use current device for properties if device is CPU
    int current_device;
    hipGetDevice(&current_device);
    int gpu = (device != hipCpuDeviceId) ? device : current_device;

    if (check_device_managed_memory(gpu)) {
      std::size_t size = len;
      if (!std::is_same<T, void>::value) {
        size = sizeof(T) * len;
      }
      
      hipError_t error = ::hipMemPrefetchAsync(src, size, device, nullptr);
      
      if (error != hipSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("hipMemPrefetchAsync(ptr={}, size={}, device={}) failed with error: {}",
                                     src, size, device, hipGetErrorString(error)));
      }
    }
  }

  template<typename T>
  static camp::resources::Event exec(T* src, int device, std::size_t len, camp::resources::Resource& r) {
    auto hip_device = r.try_get<camp::resources::Hip>();
    if (!hip_device) {
      UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Hip, got resources::{}",
                                                    platform_to_string(r.get_platform())));
    }
    auto stream = hip_device->get_stream();
    
    // Use current device for properties if device is CPU
    int current_device;
    hipGetDevice(&current_device);
    int gpu = (device != hipCpuDeviceId) ? device : current_device;

    if (check_device_managed_memory(gpu)) {
      std::size_t size = len;
      if (!std::is_same<T, void>::value) {
        size = sizeof(T) * len;
      }
      
      hipError_t error = ::hipMemPrefetchAsync(src, size, device, stream);
      
      if (error != hipSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("hipMemPrefetchAsync(ptr={}, size={}, device={}, stream={}) failed with error: {}",
                                     src, size, device, (void*)stream, hipGetErrorString(error)));
      }
    }

    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
  
  // Specializations for void*
  template<>
  static void exec<void>(void* src, int device, std::size_t len) {
    // Use current device for properties if device is CPU
    int current_device;
    hipGetDevice(&current_device);
    int gpu = (device != hipCpuDeviceId) ? device : current_device;

    if (check_device_managed_memory(gpu)) {
      hipError_t error = ::hipMemPrefetchAsync(src, len, device, nullptr);
      
      if (error != hipSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("hipMemPrefetchAsync(ptr={}, size={}, device={}) failed with error: {}",
                                     src, len, device, hipGetErrorString(error)));
      }
    }
  }
  
  template<>
  static camp::resources::Event exec<void>(void* src, int device, std::size_t len, camp::resources::Resource& r) {
    auto hip_device = r.try_get<camp::resources::Hip>();
    if (!hip_device) {
      UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Hip, got resources::{}",
                                                    platform_to_string(r.get_platform())));
    }
    auto stream = hip_device->get_stream();
    
    // Use current device for properties if device is CPU
    int current_device;
    hipGetDevice(&current_device);
    int gpu = (device != hipCpuDeviceId) ? device : current_device;

    if (check_device_managed_memory(gpu)) {
      hipError_t error = ::hipMemPrefetchAsync(src, len, device, stream);
      
      if (error != hipSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("hipMemPrefetchAsync(ptr={}, size={}, device={}, stream={}) failed with error: {}",
                                     src, len, device, (void*)stream, hipGetErrorString(error)));
      }
    }

    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
};

}
}