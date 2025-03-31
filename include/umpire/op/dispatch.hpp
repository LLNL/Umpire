#pragma once

#include "umpire/config.hpp"

#include "umpire/resource/platform.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {
namespace op {

// Base template for op_caller
template <template<typename...> class Op> 
struct op_caller {
  // Helper to get the last argument from the parameter pack
  template<typename... Args>
  static auto get_last_arg(Args... args) {
    return std::get<sizeof...(Args) - 1>(std::forward_as_tuple(args...));
  }
  
  // Helper to get the Nth argument from the parameter pack
  template<size_t N, typename... Args>
  static auto get_arg(Args... args) {
    return std::get<N>(std::forward_as_tuple(args...));
  }
  
  // Single-pointer operations (synchronous)
  template<typename T, typename... Args>
  inline static void exec(T* src, Args... args) {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();
    
    // Check for operation-specific boundary checks and event recording
    if constexpr (std::is_same_v<Op<resource::host_platform>, memset<resource::host_platform>>) {
      // For memset, the last argument is the length
      std::size_t length = get_last_arg(args...);
      
      std::ptrdiff_t offset = static_cast<char*>(src) - static_cast<char*>(src_record->ptr);
      std::size_t size = src_record->size - offset;
      
      if (length > 0 && length > size) {
        UMPIRE_ERROR(runtime_error, fmt::format("Cannot memset over the end of allocation: {} -> {}", length, size));
      }
      
      // Record the event
      umpire::event::record([&](auto& event) {
        event.name("memset")
            .category(event::category::operation)
            .arg("ptr", src)
            .arg("value", get_arg<1>(args...))  // assumes the value is the first arg
            .arg("size", length)
            .arg("allocator_ref", (void*)src_record->strategy)
            .tag("allocator_name", src_record->strategy->getName())
            .tag("replay", "true");
      });
    }
    else if constexpr (std::is_same_v<Op<resource::host_platform>, prefetch<resource::host_platform>>) {
      // For prefetch, args are: device, size
      int device = get_arg<0>(args...);
      std::size_t size = get_last_arg(args...);
      
      // Record the event
      umpire::event::record([&](auto& event) {
        event.name("prefetch")
            .category(event::category::operation)
            .arg("ptr", src)
            .arg("device", device)
            .arg("size", size)
            .arg("allocator_ref", (void*)src_record->strategy)
            .tag("allocator_name", src_record->strategy->getName())
            .tag("replay", "true");
      });
    }
    
    // Dispatch based on platform
    if (p == camp::resources::Platform::host) {
      Op<resource::host_platform>::exec(src, args...);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    else if (p == camp::resources::Platform::cuda) {
      Op<resource::cuda_platform>::exec(src, args...);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    else if (p == camp::resources::Platform::hip) {
      Op<resource::hip_platform>::exec(src, args...);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    else if (p == camp::resources::Platform::sycl) {
      Op<op::sycl_platform>::exec(src, args...);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    else if (p == camp::resources::Platform::omp_target) {
      Op<op::openmp_target_platform>::exec(src, args...);
    }
#endif
  }
  
  // Single-pointer operations (asynchronous)
  template<typename T, typename... Args>
  inline static camp::resources::EventProxy<camp::resources::Resource> 
  exec(T* src, camp::resources::Resource& ctx, Args... args) {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();
    
    // Check for operation-specific boundary checks and event recording
    if constexpr (std::is_same_v<Op<resource::host_platform>, memset<resource::host_platform>>) {
      // For memset, args are: value, length
      std::size_t length = get_last_arg(args...);
      
      std::ptrdiff_t offset = static_cast<char*>(src) - static_cast<char*>(src_record->ptr);
      std::size_t size = src_record->size - offset;
      
      if (length > 0 && length > size) {
        UMPIRE_ERROR(runtime_error, fmt::format("Cannot memset over the end of allocation: {} -> {}", length, size));
      }
      
      // Record the event
      umpire::event::record([&](auto& event) {
        event.name("memset")
            .category(event::category::operation)
            .arg("ptr", src)
            .arg("value", get_arg<0>(args...))  // assumes the value is the first arg
            .arg("size", length)
            .arg("allocator_ref", (void*)src_record->strategy)
            .tag("allocator_name", src_record->strategy->getName())
            .tag("replay", "true")
            .tag("async", "true");
      });
    }
    else if constexpr (std::is_same_v<Op<resource::host_platform>, prefetch<resource::host_platform>>) {
      // For prefetch, args are: device, size
      int device = get_arg<0>(args...);
      std::size_t size = get_last_arg(args...);
      
      // Record the event
      umpire::event::record([&](auto& event) {
        event.name("prefetch")
            .category(event::category::operation)
            .arg("ptr", src)
            .arg("device", device)
            .arg("size", size)
            .arg("allocator_ref", (void*)src_record->strategy)
            .tag("allocator_name", src_record->strategy->getName())
            .tag("replay", "true")
            .tag("async", "true");
      });
    }
    
    // Dispatch based on platform
    if (p == camp::resources::Platform::host) {
      return Op<resource::host_platform>::exec(src, args..., ctx);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    else if (p == camp::resources::Platform::cuda) {
      return Op<resource::cuda_platform>::exec(src, args..., ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    else if (p == camp::resources::Platform::hip) {
      return Op<resource::hip_platform>::exec(src, args..., ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    else if (p == camp::resources::Platform::sycl) {
      return Op<op::sycl_platform>::exec(src, args..., ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    else if (p == camp::resources::Platform::omp_target) {
      return Op<op::openmp_target_platform>::exec(src, args..., ctx);
    }
#endif

    // Fallback
    UMPIRE_ERROR(runtime_error, 
                 fmt::format("Unknown platform for operation: platform={}", 
                            static_cast<int>(p)));
    
    // Unreachable, but needed to satisfy compiler
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

  // Dual-pointer operations (synchronous)
  template<typename T, typename... Args>
  inline static void exec(T* src, T* dst, Args... args) {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(src);
    auto dst_record = allocation_map.find(dst);

    auto p1 = src_record->strategy->getPlatform();
    auto p2 = dst_record->strategy->getPlatform();
    
    // Check for operation-specific boundary checks and event recording
    if constexpr (std::is_same_v<Op<resource::host_platform, resource::host_platform>, 
                            copy<resource::host_platform, resource::host_platform>>) {
      // For copy, the last argument is the size
      std::size_t size = get_last_arg(args...);
      
      // Calculate source and destination details
      std::ptrdiff_t src_offset = static_cast<char*>(src) - static_cast<char*>(src_record->ptr);
      std::size_t src_size = src_record->size - src_offset;
      
      std::ptrdiff_t dst_offset = static_cast<char*>(dst) - static_cast<char*>(dst_record->ptr);
      std::size_t dst_size = dst_record->size - dst_offset;
      
      // If size is 0, use the source size
      if (size == 0) {
        size = src_size;
      }
      
      // Check if destination has enough space
      if (size > dst_size) {
        UMPIRE_ERROR(runtime_error,
                   fmt::format("Not enough space in destination to copy {} bytes into {} bytes", size, dst_size));
      }
      
      // Record the event
      umpire::event::record([&](auto& event) {
        event.name("copy")
            .category(event::category::operation)
            .arg("src", src)
            .arg("dst", dst)
            .arg("src_offset", src_offset)
            .arg("dst_offset", dst_offset)
            .arg("size", size)
            .arg("src_allocator_ref", (void*)src_record->strategy)
            .arg("dst_allocator_ref", (void*)dst_record->strategy)
            .tag("src_allocator_name", src_record->strategy->getName())
            .tag("dst_allocator_name", dst_record->strategy->getName())
            .tag("replay", "true");
      });
    }

    // Dispatch based on source and destination platforms
    if ((p1 == p2) && (p1 == camp::resources::Platform::host)) {
      return Op<resource::host_platform, resource::host_platform>::exec(src, dst, args...);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    if (p1 == p2 && (p1 == camp::resources::Platform::cuda)) {
      Op<resource::cuda_platform, resource::cuda_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::cuda) {
      Op<resource::host_platform, resource::cuda_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::cuda && p2 == camp::resources::Platform::host) {
      Op<resource::cuda_platform, resource::host_platform>::exec(src, dst, args...);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    if (p1 == p2 && (p1 == camp::resources::Platform::hip)) {
      Op<resource::hip_platform, resource::hip_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::hip) {
      Op<resource::host_platform, resource::hip_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::hip && p2 == camp::resources::Platform::host) {
      Op<resource::hip_platform, resource::host_platform>::exec(src, dst, args...);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    if (p1 == p2 && (p1 == camp::resources::Platform::sycl)) {
      Op<op::sycl_platform, op::sycl_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::sycl) {
      Op<resource::host_platform, op::sycl_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::sycl && p2 == camp::resources::Platform::host) {
      Op<op::sycl_platform, resource::host_platform>::exec(src, dst, args...);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    if (p1 == p2 && (p1 == camp::resources::Platform::omp_target)) {
      Op<op::openmp_target_platform, op::openmp_target_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::omp_target) {
      Op<resource::host_platform, op::openmp_target_platform>::exec(src, dst, args...);
    } else if (p1 == camp::resources::Platform::omp_target && p2 == camp::resources::Platform::host) {
      Op<op::openmp_target_platform, resource::host_platform>::exec(src, dst, args...);
    }
#endif
  }
  
  // Dual-pointer operations (asynchronous)
  template<typename T, typename... Args>
  inline static camp::resources::EventProxy<camp::resources::Resource> 
  exec(T* src, T* dst, camp::resources::Resource& ctx, Args... args) {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(src);
    auto dst_record = allocation_map.find(dst);

    auto p1 = src_record->strategy->getPlatform();
    auto p2 = dst_record->strategy->getPlatform();
    
    // Check for operation-specific boundary checks and event recording
    if constexpr (std::is_same_v<Op<resource::host_platform, resource::host_platform>, 
                            copy<resource::host_platform, resource::host_platform>>) {
      // For copy, the last argument is the size
      std::size_t size = get_last_arg(args...);
      
      // Calculate source and destination details
      std::ptrdiff_t src_offset = static_cast<char*>(src) - static_cast<char*>(src_record->ptr);
      std::size_t src_size = src_record->size - src_offset;
      
      std::ptrdiff_t dst_offset = static_cast<char*>(dst) - static_cast<char*>(dst_record->ptr);
      std::size_t dst_size = dst_record->size - dst_offset;
      
      // If size is 0, use the source size
      if (size == 0) {
        size = src_size;
      }
      
      // Check if destination has enough space
      if (size > dst_size) {
        UMPIRE_ERROR(runtime_error,
                   fmt::format("Not enough resource in destination for copy: {} -> {}", size, dst_size));
      }
      
      // Record the event
      umpire::event::record([&](auto& event) {
        event.name("copy")
            .category(event::category::operation)
            .arg("src", src)
            .arg("dst", dst)
            .arg("src_offset", src_offset)
            .arg("dst_offset", dst_offset)
            .arg("size", size)
            .arg("src_allocator_ref", (void*)src_record->strategy)
            .arg("dst_allocator_ref", (void*)dst_record->strategy)
            .tag("src_allocator_name", src_record->strategy->getName())
            .tag("dst_allocator_name", dst_record->strategy->getName())
            .tag("replay", "true")
            .tag("async", "true");
      });
    }

    // Dispatch based on source and destination platforms
    if ((p1 == p2) && (p1 == camp::resources::Platform::host)) {
      return Op<resource::host_platform, resource::host_platform>::exec(src, dst, args..., ctx);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    if (p1 == p2 && (p1 == camp::resources::Platform::cuda)) {
      return Op<resource::cuda_platform, resource::cuda_platform>::exec(src, dst, args..., ctx);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::cuda) {
      return Op<resource::host_platform, resource::cuda_platform>::exec(src, dst, args..., ctx);
    } else if (p1 == camp::resources::Platform::cuda && p2 == camp::resources::Platform::host) {
      return Op<resource::cuda_platform, resource::host_platform>::exec(src, dst, args..., ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    if (p1 == p2 && (p1 == camp::resources::Platform::hip)) {
      return Op<resource::hip_platform, resource::hip_platform>::exec(src, dst, args..., ctx);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::hip) {
      return Op<resource::host_platform, resource::hip_platform>::exec(src, dst, args..., ctx);
    } else if (p1 == camp::resources::Platform::hip && p2 == camp::resources::Platform::host) {
      return Op<resource::hip_platform, resource::host_platform>::exec(src, dst, args..., ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    if (p1 == p2 && (p1 == camp::resources::Platform::sycl)) {
      return Op<op::sycl_platform, op::sycl_platform>::exec(src, dst, args..., ctx);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::sycl) {
      return Op<resource::host_platform, op::sycl_platform>::exec(src, dst, args..., ctx);
    } else if (p1 == camp::resources::Platform::sycl && p2 == camp::resources::Platform::host) {
      return Op<op::sycl_platform, resource::host_platform>::exec(src, dst, args..., ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    if (p1 == p2 && (p1 == camp::resources::Platform::omp_target)) {
      return Op<op::openmp_target_platform, op::openmp_target_platform>::exec(src, dst, args..., ctx);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::omp_target) {
      return Op<resource::host_platform, op::openmp_target_platform>::exec(src, dst, args..., ctx);
    } else if (p1 == camp::resources::Platform::omp_target && p2 == camp::resources::Platform::host) {
      return Op<op::openmp_target_platform, resource::host_platform>::exec(src, dst, args..., ctx);
    }
#endif

    // Fallback
    UMPIRE_ERROR(runtime_error, 
                 fmt::format("Unknown platforms for operation: src_platform={}, dst_platform={}", 
                            static_cast<int>(p1), static_cast<int>(p2)));
    
    // Unreachable, but needed to satisfy compiler
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }
};

}

template <typename T>
void copy(T* src, T* dst, std::size_t len) {
    op::op_caller<op::copy>::exec(src, dst, len);
}

template <typename T>
camp::resources::EventProxy<camp::resources::Resource> copy(T* src, T* dst, camp::resources::Resource& ctx, std::size_t len) {
    return op::op_caller<op::copy>::exec(src, dst, ctx, len);
}

template <typename T, typename V>
void memset(T* src, V v, std::size_t len) {
    op::op_caller<op::memset>::exec(src, v, len);
}

template <typename T>
camp::resources::EventProxy<camp::resources::Resource> memset(T* src, int v, camp::resources::Resource& ctx, std::size_t len) {
    return op::op_caller<op::memset>::exec(src, ctx, v, len);
}

template <typename T>
T* reallocate(T* src, std::size_t size) {
    // We'll handle the void* case as a specialization
    
    if (src == nullptr) {
        // If src is nullptr, just allocate memory from the default allocator
        auto& rm = ResourceManager::getInstance();
        Allocator allocator = rm.getDefaultAllocator();
        return static_cast<T*>(allocator.allocate(size * sizeof(T)));
    }
    
    // Otherwise, use the platform-specific implementation if available,
    // falling back to the generic implementation
    auto& allocation_map = ResourceManager::getInstance().m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();
    
    T* new_ptr = nullptr;
    
    if (p == camp::resources::Platform::host) {
      new_ptr = op::generic_reallocate<resource::host_platform>::exec(src, size);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    else if (p == camp::resources::Platform::cuda) {
      new_ptr = op::generic_reallocate<resource::cuda_platform>::exec(src, size);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    else if (p == camp::resources::Platform::hip) {
      new_ptr = op::generic_reallocate<resource::hip_platform>::exec(src, size);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    else if (p == camp::resources::Platform::sycl) {
      new_ptr = op::generic_reallocate<op::sycl_platform>::exec(src, size);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    else if (p == camp::resources::Platform::omp_target) {
      new_ptr = op::generic_reallocate<op::openmp_target_platform>::exec(src, size);
    }
#endif
    else {
      // Fallback to generic implementation
      new_ptr = op::generic_reallocate<resource::undefined_platform>::exec(src, size);
    }
    
    return new_ptr;
}

// Explicit specialization for void*
template<>
void* reallocate(void* src, std::size_t size) {
    if (src == nullptr) {
        // If src is nullptr, just allocate memory from the default allocator
        auto& rm = ResourceManager::getInstance();
        Allocator allocator = rm.getDefaultAllocator();
        return allocator.allocate(size);
    }
    
    // Otherwise, use the platform-specific implementation if available,
    // falling back to the generic implementation
    auto& allocation_map = ResourceManager::getInstance().m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();
    
    void* new_ptr = nullptr;
    
    if (p == camp::resources::Platform::host) {
      new_ptr = op::generic_reallocate<resource::host_platform>::exec(src, size);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    else if (p == camp::resources::Platform::cuda) {
      new_ptr = op::generic_reallocate<resource::cuda_platform>::exec(src, size);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    else if (p == camp::resources::Platform::hip) {
      new_ptr = op::generic_reallocate<resource::hip_platform>::exec(src, size);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    else if (p == camp::resources::Platform::sycl) {
      new_ptr = op::generic_reallocate<op::sycl_platform>::exec(src, size);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    else if (p == camp::resources::Platform::omp_target) {
      new_ptr = op::generic_reallocate<op::openmp_target_platform>::exec(src, size);
    }
#endif
    else {
      // Fallback to generic implementation
      new_ptr = op::generic_reallocate<resource::undefined_platform>::exec(src, size);
    }
    
    return new_ptr;
}

template <typename T>
camp::resources::EventProxy<camp::resources::Resource> reallocate(T* src, std::size_t size, camp::resources::Resource& ctx) {
    
    if (src == nullptr) {
        // If src is nullptr, just allocate memory from the default allocator
        auto& rm = ResourceManager::getInstance();
        Allocator allocator = rm.getDefaultAllocator();
        allocator.allocate(size * sizeof(T));
        return camp::resources::EventProxy<camp::resources::Resource>{ctx};
    }
    
    // Otherwise, use the platform-specific implementation if available,
    // falling back to the generic implementation
    auto& allocation_map = ResourceManager::getInstance().m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();
    
    if (p == camp::resources::Platform::host) {
      return op::generic_reallocate<resource::host_platform>::exec(src, size, ctx);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    else if (p == camp::resources::Platform::cuda) {
      return op::generic_reallocate<resource::cuda_platform>::exec(src, size, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    else if (p == camp::resources::Platform::hip) {
      return op::generic_reallocate<resource::hip_platform>::exec(src, size, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    else if (p == camp::resources::Platform::sycl) {
      return op::generic_reallocate<op::sycl_platform>::exec(src, size, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    else if (p == camp::resources::Platform::omp_target) {
      return op::generic_reallocate<op::openmp_target_platform>::exec(src, size, ctx);
    }
#endif
    
    // Fallback to generic implementation
    return op::generic_reallocate<resource::undefined_platform>::exec(src, size, ctx);
}

// Explicit specialization for void*
template<>
camp::resources::EventProxy<camp::resources::Resource> reallocate(void* src, std::size_t size, camp::resources::Resource& ctx) {
    if (src == nullptr) {
        // If src is nullptr, just allocate memory from the default allocator
        auto& rm = ResourceManager::getInstance();
        Allocator allocator = rm.getDefaultAllocator();
        allocator.allocate(size);
        return camp::resources::EventProxy<camp::resources::Resource>{ctx};
    }
    
    // Otherwise, use the platform-specific implementation if available,
    // falling back to the generic implementation
    auto& allocation_map = ResourceManager::getInstance().m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();
    
    if (p == camp::resources::Platform::host) {
      return op::generic_reallocate<resource::host_platform>::exec(src, size, ctx);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    else if (p == camp::resources::Platform::cuda) {
      return op::generic_reallocate<resource::cuda_platform>::exec(src, size, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    else if (p == camp::resources::Platform::hip) {
      return op::generic_reallocate<resource::hip_platform>::exec(src, size, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    else if (p == camp::resources::Platform::sycl) {
      return op::generic_reallocate<op::sycl_platform>::exec(src, size, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    else if (p == camp::resources::Platform::omp_target) {
      return op::generic_reallocate<op::openmp_target_platform>::exec(src, size, ctx);
    }
#endif
    
    // Fallback to generic implementation
    return op::generic_reallocate<resource::undefined_platform>::exec(src, size, ctx);
}

template <typename T>
camp::resources::EventProxy<camp::resources::Resource> prefetch(T* ptr, int device, camp::resources::Resource& ctx, std::size_t size) {
    return op::op_caller<op::prefetch>::exec(ptr, ctx, device, size);
}

}