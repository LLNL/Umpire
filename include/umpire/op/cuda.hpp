#pragma once

#include "umpire/resource/platform.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/Platform.hpp"

// Forward declaration of kernel for launching directly in device code if needed
extern "C" {
__global__ void
umpire_cuda_fill(void* data, int value, std::size_t length);
}

namespace {
  template<typename SRC, typename DST>
  struct get_kind;

  template<>
  struct get_kind<resource::cuda_platform, resource::host_platform> {
    static constexpr cudaMemcpyKind value = cudaMemcpyDeviceToHost;
  };

  template<>
  struct get_kind<resource::host_platform, resource::cuda_platform> {
    static constexpr cudaMemcpyKind value = cudaMemcpyHostToDevice;
  };

  template<>
  struct get_kind<resource::cuda_platform, resource::cuda_platform> {
    static constexpr cudaMemcpyKind value = cudaMemcpyDeviceToDevice;
  };
}

namespace umpire {
namespace op {

namespace {
  // Helper function to check if a CUDA device supports managed memory features
  inline bool check_device_managed_memory(int device) {
    cudaDeviceProp properties;
    cudaError_t error = ::cudaGetDeviceProperties(&properties, device);
    
    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("cudaGetDeviceProperties for device {} failed with error: {}",
                                   device, cudaGetErrorString(error)));
    }
    
    return (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1);
  }
  
  // Generic function to handle cudaMemAdvise operations
  template<typename T>
  inline void advise_impl(T* ptr, std::size_t n, int device, cudaMemoryAdvise advice) {
    std::size_t size = n;
    if (std::is_same<T, void>::value) {
      // void pointers don't have a size
    } else {
      size = sizeof(T) * n;
    }
    
    if (check_device_managed_memory(device)) {
      cudaError_t error = ::cudaMemAdvise(ptr, size, advice, device);

      if (error != cudaSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("cudaMemAdvise(ptr={}, size={}, advice={}, device={}) failed with error: {}",
                                     ptr, size, static_cast<int>(advice), device, cudaGetErrorString(error)));
      }
    }
  }

  // Generic copy function that handles the different kinds of copies
  template<typename T>
  inline void copy_impl(T* src, T* dst, std::size_t len, cudaMemcpyKind kind) {
    std::size_t size = len;
    if (!std::is_same<T, void>::value) {
      size = sizeof(T) * len;
    }
    
    cudaError_t error = ::cudaMemcpy(dst, src, size, kind);
    if (error != cudaSuccess) {
      UMPIRE_ERROR(
          runtime_error,
          umpire::fmt::format(
              "cudaMemcpy(dst={}, src={}, size={}, kind={}) failed with error: {}",
              dst, src, size, static_cast<int>(kind), cudaGetErrorString(error)));
    }
  }

  // Async version of copy for use with CUDA streams
  template<typename T>
  inline camp::resources::EventProxy<camp::resources::Resource> copy_async_impl(T* src, T* dst, std::size_t len, camp::resources::Resource& r, cudaMemcpyKind kind) {
    auto device = r.try_get<camp::resources::Cuda>();
    if (!device) {
      UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Cuda, got resources::{}",
                                                    platform_to_string(r.get_platform())));
    }
    auto stream = device->get_stream();

    std::size_t size = len;
    if (!std::is_same<T, void>::value) {
      size = sizeof(T) * len;
    }

    cudaError_t error = ::cudaMemcpyAsync(dst, src, size, kind, stream);

    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("cudaMemcpyAsync(dst={}, src={}, size={}, kind={}, stream={}) failed with error: {}",
                                    dst, src, size, static_cast<int>(kind), 
                                    (void*)stream, cudaGetErrorString(error)));
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
    
    cudaError_t error = ::cudaMemset(ptr, value, size);

    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("cudaMemset(ptr={}, value={}, size={}) failed with error: {}", 
                                    ptr, value, size, cudaGetErrorString(error)));
    }
  }
  
  // Async version of memset
  template<typename T>
  inline camp::resources::EventProxy<camp::resources::Resource> memset_async_impl(T* ptr, int value, std::size_t len, camp::resources::Resource& r) {
    auto device = r.try_get<camp::resources::Cuda>();
    if (!device) {
      UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Cuda, got resources::{}",
                                                    platform_to_string(r.get_platform())));
    }
    auto stream = device->get_stream();

    std::size_t size = len;
    if (!std::is_same<T, void>::value) {
      size = sizeof(T) * len;
    }

    cudaError_t error = ::cudaMemsetAsync(ptr, value, size, stream);

    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format(
                    "cudaMemsetAsync(ptr={}, value={}, size={}, stream={}) failed with error: {}",
                    ptr, value, size, (void*)stream, cudaGetErrorString(error)));
    }

    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
}

// Copy operations for different platform combinations
template<>
struct copy<resource::cuda_platform, resource::cuda_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len, cudaMemcpyDeviceToDevice);
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    return copy_async_impl(src, dst, len, r, cudaMemcpyDeviceToDevice);
  }
};

template<>
struct copy<resource::cuda_platform, resource::host_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len, cudaMemcpyDeviceToHost);
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    return copy_async_impl(src, dst, len, r, cudaMemcpyDeviceToHost);
  }
};

template<>
struct copy<resource::host_platform, resource::cuda_platform>
{
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len, cudaMemcpyHostToDevice);
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    return copy_async_impl(src, dst, len, r, cudaMemcpyHostToDevice);
  }
};

// Special handling for void pointers
template<>
template<>
inline void copy<resource::cuda_platform, resource::cuda_platform>::exec<void>(void* src, void* dst, std::size_t len) {
  copy_impl<void>(src, dst, len, cudaMemcpyDeviceToDevice);
}

template<>
template<>
inline void copy<resource::cuda_platform, resource::host_platform>::exec<void>(void* src, void* dst, std::size_t len) {
  copy_impl<void>(src, dst, len, cudaMemcpyDeviceToHost);
}

template<>
template<>
inline void copy<resource::host_platform, resource::cuda_platform>::exec<void>(void* src, void* dst, std::size_t len) {
  copy_impl<void>(src, dst, len, cudaMemcpyHostToDevice);
}

// Memset operations
template<>
struct memset<resource::cuda_platform>
{
  template<typename T>
  static void exec(T* src, int val, std::size_t len) {
    memset_impl(src, val, len);
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* src, int val, std::size_t len, camp::resources::Resource& r) {
    return memset_async_impl(src, val, len, r);
  }
  
  // Specialization for void*
  template<>
  static void exec<void>(void* src, int val, std::size_t len) {
    memset_impl<void>(src, val, len);
  }
  
  template<>
  static camp::resources::EventProxy<camp::resources::Resource> exec<void>(void* src, int val, std::size_t len, camp::resources::Resource& r) {
    return memset_async_impl<void>(src, val, len, r);
  }
};

// Reallocate implementation
template<>
struct reallocate<resource::cuda_platform>
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
struct accessed_by<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, cudaMemAdviseSetAccessedBy);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, cudaMemAdviseSetAccessedBy);
  }
};

template<>
struct preferred_location<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, cudaMemAdviseSetPreferredLocation);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, cudaMemAdviseSetPreferredLocation);
  }
};

template<>
struct read_mostly<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, cudaMemAdviseSetReadMostly);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, cudaMemAdviseSetReadMostly);
  }
};

template<>
struct unset_accessed_by<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, cudaMemAdviseUnsetAccessedBy);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, cudaMemAdviseUnsetAccessedBy);
  }
};

template<>
struct unset_preferred_location<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, cudaMemAdviseUnsetPreferredLocation);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, cudaMemAdviseUnsetPreferredLocation);
  }
};

template<>
struct unset_read_mostly<resource::cuda_platform>
{
  template <typename T>
  static inline void exec(T* src, int device, std::size_t len) {
    advise_impl(src, len, device, cudaMemAdviseUnsetReadMostly);
  }
  
  template<>
  static inline void exec<void>(void* src, int device, std::size_t len) {
    advise_impl<void>(src, len, device, cudaMemAdviseUnsetReadMostly);
  }
};

// Prefetch operations
template<>
struct prefetch<resource::cuda_platform>
{
  template<typename T>
  static void exec(T* src, int device, std::size_t len) {
    // Use current device for properties if device is CPU
    int current_device;
    cudaGetDevice(&current_device);
    int gpu = (device != cudaCpuDeviceId) ? device : current_device;

    if (check_device_managed_memory(gpu)) {
      std::size_t size = len;
      if (!std::is_same<T, void>::value) {
        size = sizeof(T) * len;
      }
      
      cudaError_t error = ::cudaMemPrefetchAsync(src, size, device, nullptr);
      
      if (error != cudaSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("cudaMemPrefetchAsync(ptr={}, size={}, device={}) failed with error: {}",
                                     src, size, device, cudaGetErrorString(error)));
      }
    }
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(T* src, int device, std::size_t len, camp::resources::Resource& r) {
    auto cuda_device = r.try_get<camp::resources::Cuda>();
    if (!cuda_device) {
      UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Cuda, got resources::{}",
                                                    platform_to_string(r.get_platform())));
    }
    auto stream = cuda_device->get_stream();
    
    // Use current device for properties if device is CPU
    int current_device;
    cudaGetDevice(&current_device);
    int gpu = (device != cudaCpuDeviceId) ? device : current_device;

    if (check_device_managed_memory(gpu)) {
      std::size_t size = len;
      if (!std::is_same<T, void>::value) {
        size = sizeof(T) * len;
      }
      
      cudaError_t error = ::cudaMemPrefetchAsync(src, size, device, stream);
      
      if (error != cudaSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("cudaMemPrefetchAsync(ptr={}, size={}, device={}, stream={}) failed with error: {}",
                                     src, size, device, (void*)stream, cudaGetErrorString(error)));
      }
    }

    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
  
  // Specializations for void*
  template<>
  static void exec<void>(void* src, int device, std::size_t len) {
    // Use current device for properties if device is CPU
    int current_device;
    cudaGetDevice(&current_device);
    int gpu = (device != cudaCpuDeviceId) ? device : current_device;

    if (check_device_managed_memory(gpu)) {
      cudaError_t error = ::cudaMemPrefetchAsync(src, len, device, nullptr);
      
      if (error != cudaSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("cudaMemPrefetchAsync(ptr={}, size={}, device={}) failed with error: {}",
                                     src, len, device, cudaGetErrorString(error)));
      }
    }
  }
  
  template<>
  static camp::resources::EventProxy<camp::resources::Resource> exec<void>(void* src, int device, std::size_t len, camp::resources::Resource& r) {
    auto cuda_device = r.try_get<camp::resources::Cuda>();
    if (!cuda_device) {
      UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Cuda, got resources::{}",
                                                    platform_to_string(r.get_platform())));
    }
    auto stream = cuda_device->get_stream();
    
    // Use current device for properties if device is CPU
    int current_device;
    cudaGetDevice(&current_device);
    int gpu = (device != cudaCpuDeviceId) ? device : current_device;

    if (check_device_managed_memory(gpu)) {
      cudaError_t error = ::cudaMemPrefetchAsync(src, len, device, stream);
      
      if (error != cudaSuccess) {
        UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("cudaMemPrefetchAsync(ptr={}, size={}, device={}, stream={}) failed with error: {}",
                                     src, len, device, (void*)stream, cudaGetErrorString(error)));
      }
    }

    return camp::resources::EventProxy<camp::resources::Resource>{r};
  }
};

}
}