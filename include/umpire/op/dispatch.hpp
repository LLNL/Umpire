#pragma once

#include "umpire/config.hpp"

#include "umpire/resource/platform.hpp"

#include "umpire/ResourceManager.hpp"

namespace umpire {
namespace op {


template <int args, template<typename... T> class Op> struct op_caller{};

template<template<typename T> class Op>
struct op_caller<1, Op > {
  template<typename T, typename... Args>
  inline static void exec(T* src, Args... args) {
    auto& allocation_map = ResourceManager::getInstance().m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();
    
    // get src platform
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
};

template<class... Ts>
struct count {
    static constexpr std::size_t value = sizeof...(Ts);
};


template<template<typename... Ts> class Op>
struct op_caller<2, Op> {
  // try calling with Op::arity
  template<typename T, typename... Args>
  inline static void exec(T* src, T* dst, Args... args) {
    auto& allocation_map = ResourceManager::getInstance().m_allocations;
    auto src_record = allocation_map.find(src);
    auto dst_record = allocation_map.find(dst);

    auto p1 = src_record->strategy->getPlatform();
    auto p2 = dst_record->strategy->getPlatform();

    // get src and dest platform
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
};

}

// template<typename Src, typename Dst, typename T>
// void copy(T* src, T* dst, std::size_t len) {
//   op::copy<typename Src::platform, typename Dst::platform>::exec(src, dst, len);
// }

template <typename T>
void copy(T* src, T* dst, std::size_t len) {
    op::op_caller<2, op::copy>::exec(src, dst, len);
}

template <typename T>
camp::resources::EventProxy<camp::resources::Resource> copy(T* src, T* dst, camp::resources::Resource& ctx, std::size_t len) {
    auto& allocation_map = ResourceManager::getInstance().m_allocations;
    auto src_record = allocation_map.find(src);
    auto dst_record = allocation_map.find(dst);

    auto p1 = src_record->strategy->getPlatform();
    auto p2 = dst_record->strategy->getPlatform();
    
    // get src and dest platform
    if ((p1 == p2) && (p1 == camp::resources::Platform::host)) {
      return op::copy<resource::host_platform, resource::host_platform>::exec(src, dst, len, ctx);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    if (p1 == p2 && (p1 == camp::resources::Platform::cuda)) {
      return op::copy<resource::cuda_platform, resource::cuda_platform>::exec(src, dst, len, ctx);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::cuda) {
      return op::copy<resource::host_platform, resource::cuda_platform>::exec(src, dst, len, ctx);
    } else if (p1 == camp::resources::Platform::cuda && p2 == camp::resources::Platform::host) {
      return op::copy<resource::cuda_platform, resource::host_platform>::exec(src, dst, len, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    if (p1 == p2 && (p1 == camp::resources::Platform::hip)) {
      return op::copy<resource::hip_platform, resource::hip_platform>::exec(src, dst, len, ctx);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::hip) {
      return op::copy<resource::host_platform, resource::hip_platform>::exec(src, dst, len, ctx);
    } else if (p1 == camp::resources::Platform::hip && p2 == camp::resources::Platform::host) {
      return op::copy<resource::hip_platform, resource::host_platform>::exec(src, dst, len, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    if (p1 == p2 && (p1 == camp::resources::Platform::sycl)) {
      return op::copy<op::sycl_platform, op::sycl_platform>::exec(src, dst, len, ctx);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::sycl) {
      return op::copy<resource::host_platform, op::sycl_platform>::exec(src, dst, len, ctx);
    } else if (p1 == camp::resources::Platform::sycl && p2 == camp::resources::Platform::host) {
      return op::copy<op::sycl_platform, resource::host_platform>::exec(src, dst, len, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    if (p1 == p2 && (p1 == camp::resources::Platform::omp_target)) {
      return op::copy<op::openmp_target_platform, op::openmp_target_platform>::exec(src, dst, len, ctx);
    } else if (p1 == camp::resources::Platform::host && p2 == camp::resources::Platform::omp_target) {
      return op::copy<resource::host_platform, op::openmp_target_platform>::exec(src, dst, len, ctx);
    } else if (p1 == camp::resources::Platform::omp_target && p2 == camp::resources::Platform::host) {
      return op::copy<op::openmp_target_platform, resource::host_platform>::exec(src, dst, len, ctx);
    }
#endif
    
    UMPIRE_ERROR(runtime_error, 
                 fmt::format("Unknown platforms for copy: src={}, dst={}", 
                            static_cast<int>(p1), static_cast<int>(p2)));
    
    // Unreachable, but needed to satisfy compiler
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

template <typename T, typename V>
void memset(T* src, V v, std::size_t len) {
    op::op_caller<1, op::memset>::exec(src, v, len);
}

template <typename T>
camp::resources::EventProxy<camp::resources::Resource> memset(T* src, int v, camp::resources::Resource& ctx, std::size_t len) {
    auto& allocation_map = ResourceManager::getInstance().m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();
    
    if (p == camp::resources::Platform::host) {
      return op::memset<resource::host_platform>::exec(src, v, len, ctx);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    else if (p == camp::resources::Platform::cuda) {
      return op::memset<resource::cuda_platform>::exec(src, v, len, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    else if (p == camp::resources::Platform::hip) {
      return op::memset<resource::hip_platform>::exec(src, v, len, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    else if (p == camp::resources::Platform::sycl) {
      return op::memset<op::sycl_platform>::exec(src, v, len, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    else if (p == camp::resources::Platform::omp_target) {
      return op::memset<op::openmp_target_platform>::exec(src, v, len, ctx);
    }
#endif
    
    UMPIRE_ERROR(runtime_error, 
                 fmt::format("Unknown platform for memset: platform={}", 
                            static_cast<int>(p)));
    
    // Unreachable, but needed to satisfy compiler
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

template <typename T>
T* reallocate(T* src, std::size_t size) {
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
    
    if (p == camp::resources::Platform::host) {
      return op::generic_reallocate<resource::host_platform>::exec(src, size);
    } 
#if defined(UMPIRE_ENABLE_CUDA)
    else if (p == camp::resources::Platform::cuda) {
      return op::generic_reallocate<resource::cuda_platform>::exec(src, size);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    else if (p == camp::resources::Platform::hip) {
      return op::generic_reallocate<resource::hip_platform>::exec(src, size);
    }
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    else if (p == camp::resources::Platform::sycl) {
      return op::generic_reallocate<op::sycl_platform>::exec(src, size);
    }
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    else if (p == camp::resources::Platform::omp_target) {
      return op::generic_reallocate<op::openmp_target_platform>::exec(src, size);
    }
#endif
    
    // Fallback to generic implementation
    return op::generic_reallocate<resource::undefined_platform>::exec(src, size);
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

template <typename T>
camp::resources::EventProxy<camp::resources::Resource> prefetch(T* ptr, int device, camp::resources::Resource& ctx, std::size_t size) {
    auto& allocation_map = ResourceManager::getInstance().m_allocations;
    auto ptr_record = allocation_map.find(ptr);
    auto p = ptr_record->strategy->getPlatform();
    
    // Currently only CUDA and HIP platforms support prefetch
#if defined(UMPIRE_ENABLE_CUDA)
    if (p == camp::resources::Platform::cuda) {
      return op::prefetch<resource::cuda_platform>::exec(ptr, device, size, ctx);
    }
#endif
#if defined(UMPIRE_ENABLE_HIP)
    if (p == camp::resources::Platform::hip) {
      return op::prefetch<resource::hip_platform>::exec(ptr, device, size, ctx);
    }
#endif
    
    UMPIRE_ERROR(runtime_error, 
                 fmt::format("Prefetch not supported for platform: {}", 
                            static_cast<int>(p)));
    
    // Unreachable, but needed to satisfy compiler
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

}