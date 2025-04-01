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
// Copy direction mapping via template specialization
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

// HIP implementation helpers
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

// Size-aware calculation with type awareness
template<typename T>
inline std::size_t calculate_size(T* ptr, std::size_t count) {
  return std::is_same<T, void>::value ? count : count * sizeof(T);
}

// Memory advice operation helper
template<typename T>
inline void advise_impl(T* ptr, std::size_t count, int device, hipMemoryAdvise advice) {
  if (!check_device_managed_memory(device)) return;
  
  std::size_t size = calculate_size(ptr, count);
  hipError_t error = ::hipMemAdvise(ptr, size, advice, device);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
               umpire::fmt::format("hipMemAdvise(ptr={}, size={}, advice={}, device={}) failed with error: {}",
                                 ptr, size, static_cast<int>(advice), device, hipGetErrorString(error)));
  }
}

// HIP copy implementation
template<typename T>
inline void copy_impl(T* src, T* dst, std::size_t count, hipMemcpyKind kind) {
  std::size_t size = calculate_size(src, count);
  
  hipError_t error = ::hipMemcpy(dst, src, size, kind);
  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
        umpire::fmt::format("hipMemcpy(dst={}, src={}, size={}, kind={}) failed with error: {}",
            dst, src, size, static_cast<int>(kind), hipGetErrorString(error)));
  }
}

// HIP async copy implementation
template<typename T>
inline camp::resources::EventProxy<camp::resources::Resource> copy_async_impl(
    T* src, T* dst, std::size_t count, camp::resources::Resource& r, hipMemcpyKind kind) {
  
  auto device = r.try_get<camp::resources::Hip>();
  if (!device) {
    UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Hip, got resources::{}",
                                                  platform_to_string(r.get_platform())));
  }
  auto stream = device->get_stream();
  std::size_t size = calculate_size(src, count);

  hipError_t error = ::hipMemcpyAsync(dst, src, size, kind, stream);
  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
               umpire::fmt::format("hipMemcpyAsync(dst={}, src={}, size={}, kind={}, stream={}) failed with error: {}",
                                  dst, src, size, static_cast<int>(kind), 
                                  (void*)stream, hipGetErrorString(error)));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{r};
}

// HIP memset implementation
template<typename T>
inline void memset_impl(T* ptr, int value, std::size_t count) {
  std::size_t size = calculate_size(ptr, count);
  
  hipError_t error = ::hipMemset(ptr, value, size);
  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
               umpire::fmt::format("hipMemset(ptr={}, value={}, size={}) failed with error: {}", 
                                  ptr, value, size, hipGetErrorString(error)));
  }
}

// HIP async memset implementation
template<typename T>
inline camp::resources::EventProxy<camp::resources::Resource> memset_async_impl(
    T* ptr, int value, std::size_t count, camp::resources::Resource& r) {
  
  auto device = r.try_get<camp::resources::Hip>();
  if (!device) {
    UMPIRE_ERROR(resource_error, umpire::fmt::format("Expected resources::Hip, got resources::{}",
                                                  platform_to_string(r.get_platform())));
  }
  auto stream = device->get_stream();
  std::size_t size = calculate_size(ptr, count);

  hipError_t error = ::hipMemsetAsync(ptr, value, size, stream);
  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
               umpire::fmt::format(
                  "hipMemsetAsync(ptr={}, value={}, size={}, stream={}) failed with error: {}",
                  ptr, value, size, (void*)stream, hipGetErrorString(error)));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{r};
}

// Prefetch implementation
template<typename T>
inline void prefetch_impl(T* ptr, int device, std::size_t count) {
  // Use current device for properties if device is CPU
  int current_device;
  hipGetDevice(&current_device);
  int gpu = (device != hipCpuDeviceId) ? device : current_device;

  if (check_device_managed_memory(gpu)) {
    std::size_t size = calculate_size(ptr, count);
    hipError_t error = ::hipMemPrefetchAsync(ptr, size, device, nullptr);
    
    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("hipMemPrefetchAsync(ptr={}, size={}, device={}) failed with error: {}",
                                   ptr, size, device, hipGetErrorString(error)));
    }
  }
}

// Async prefetch implementation
template<typename T>
inline camp::resources::EventProxy<camp::resources::Resource> prefetch_async_impl(
    T* ptr, int device, std::size_t count, camp::resources::Resource& r) {
  
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
    std::size_t size = calculate_size(ptr, count);
    hipError_t error = ::hipMemPrefetchAsync(ptr, size, device, stream);
    
    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("hipMemPrefetchAsync(ptr={}, size={}, device={}, stream={}) failed with error: {}",
                                   ptr, size, device, (void*)stream, hipGetErrorString(error)));
    }
  }

  return camp::resources::EventProxy<camp::resources::Resource>{r};
}
} // namespace

// Copy operations for different platform combinations
template<>
struct copy<resource::hip_platform, resource::hip_platform> {
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len, hipMemcpyDeviceToDevice);
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    return copy_async_impl(src, dst, len, r, hipMemcpyDeviceToDevice);
  }
};

template<>
struct copy<resource::hip_platform, resource::host_platform> {
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len, hipMemcpyDeviceToHost);
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    return copy_async_impl(src, dst, len, r, hipMemcpyDeviceToHost);
  }
};

template<>
struct copy<resource::host_platform, resource::hip_platform> {
  template<typename T>
  static void exec(T* src, T* dst, std::size_t len) {
    copy_impl(src, dst, len, hipMemcpyHostToDevice);
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src, T* dst, std::size_t len, camp::resources::Resource& r) {
    return copy_async_impl(src, dst, len, r, hipMemcpyHostToDevice);
  }
};

// Memset operations
template<>
struct memset<resource::hip_platform> {
  template<typename T>
  static void exec(T* src, int val, std::size_t len) {
    memset_impl(src, val, len);
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src, int val, std::size_t len, camp::resources::Resource& r) {
    return memset_async_impl(src, val, len, r);
  }
};

// Reallocate implementation - stub only
template<>
struct reallocate<resource::hip_platform> {
  template<typename T>
  static T* exec(T* src, std::size_t size) {
    // HIP doesn't have direct realloc, this needs to be handled at ResourceManager level
    return nullptr;
  }
};

// Memory advice operations - using macro to reduce duplication
#define DEFINE_HIP_ADVICE_OP(op_name, advice_flag) \
template<> \
struct op_name<resource::hip_platform> { \
  template <typename T> \
  static inline void exec(T* src, int device, std::size_t len) { \
    advise_impl(src, len, device, advice_flag); \
  } \
};

DEFINE_HIP_ADVICE_OP(accessed_by, hipMemAdviseSetAccessedBy)
DEFINE_HIP_ADVICE_OP(preferred_location, hipMemAdviseSetPreferredLocation)
DEFINE_HIP_ADVICE_OP(read_mostly, hipMemAdviseSetReadMostly)
DEFINE_HIP_ADVICE_OP(unset_accessed_by, hipMemAdviseUnsetAccessedBy)
DEFINE_HIP_ADVICE_OP(unset_preferred_location, hipMemAdviseUnsetPreferredLocation)
DEFINE_HIP_ADVICE_OP(unset_read_mostly, hipMemAdviseUnsetReadMostly)

#if HIP_VERSION_MAJOR >= 5
DEFINE_HIP_ADVICE_OP(coarse_grain, hipMemAdviseSetCoarseGrain)
DEFINE_HIP_ADVICE_OP(unset_coarse_grain, hipMemAdviseUnsetCoarseGrain)
#endif

#undef DEFINE_HIP_ADVICE_OP

// Prefetch operations
template<>
struct prefetch<resource::hip_platform> {
  template<typename T>
  static void exec(T* src, int device, std::size_t len) {
    prefetch_impl(src, device, len);
  }

  template<typename T>
  static camp::resources::EventProxy<camp::resources::Resource> exec(
      T* src, int device, std::size_t len, camp::resources::Resource& r) {
    return prefetch_async_impl(src, device, len, r);
  }
};

} // namespace op
} // namespace umpire