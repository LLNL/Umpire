#pragma once

#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/platform.hpp"

namespace umpire {
namespace op {

// Platform dispatch for single-platform operations
template <template <typename...> class Op, typename... Args>
inline auto dispatch_by_platform(camp::resources::Platform platform, Args&&... args)
{
  switch (platform) {
    case camp::resources::Platform::host:
      return Op<resource::host_platform>::exec(std::forward<Args>(args)...);
#if defined(UMPIRE_ENABLE_CUDA)
    case camp::resources::Platform::cuda:
      return Op<resource::cuda_platform>::exec(std::forward<Args>(args)...);
#endif
#if defined(UMPIRE_ENABLE_HIP)
    case camp::resources::Platform::hip:
      return Op<resource::hip_platform>::exec(std::forward<Args>(args)...);
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    case camp::resources::Platform::sycl:
      return Op<resource::sycl_platform>::exec(std::forward<Args>(args)...);
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    case camp::resources::Platform::omp_target:
      return Op<resource::openmp_target_platform>::exec(std::forward<Args>(args)...);
#endif
    default:
      UMPIRE_ERROR(runtime_error, "Unknown platform for operation");
  }
}

// Platform dispatch for dual-platform operations
template <template <typename...> class Op, typename... Args>
inline auto dispatch_dual_platforms(camp::resources::Platform src_platform, camp::resources::Platform dst_platform,
                                    Args&&... args)
{
  // Same-platform operations
  if (src_platform == dst_platform) {
    switch (src_platform) {
      case camp::resources::Platform::host:
        return Op<resource::host_platform, resource::host_platform>::exec(std::forward<Args>(args)...);
#if defined(UMPIRE_ENABLE_CUDA)
      case camp::resources::Platform::cuda:
        return Op<resource::cuda_platform, resource::cuda_platform>::exec(std::forward<Args>(args)...);
#endif
#if defined(UMPIRE_ENABLE_HIP)
      case camp::resources::Platform::hip:
        return Op<resource::hip_platform, resource::hip_platform>::exec(std::forward<Args>(args)...);
#endif
#if defined(UMPIRE_ENABLE_SYCL)
      case camp::resources::Platform::sycl:
        return Op<resource::sycl_platform, resource::sycl_platform>::exec(std::forward<Args>(args)...);
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
      case camp::resources::Platform::omp_target:
        return Op<resource::openmp_target_platform, resource::openmp_target_platform>::exec(
            std::forward<Args>(args)...);
#endif
      default:
        UMPIRE_ERROR(runtime_error, "Unknown platform for same-platform operation");
    }
  }

  // Cross-platform operations
#if defined(UMPIRE_ENABLE_CUDA)
  if (src_platform == camp::resources::Platform::host && dst_platform == camp::resources::Platform::cuda) {
    return Op<resource::host_platform, resource::cuda_platform>::exec(std::forward<Args>(args)...);
  }
  if (src_platform == camp::resources::Platform::cuda && dst_platform == camp::resources::Platform::host) {
    return Op<resource::cuda_platform, resource::host_platform>::exec(std::forward<Args>(args)...);
  }
#endif

#if defined(UMPIRE_ENABLE_HIP)
  if (src_platform == camp::resources::Platform::host && dst_platform == camp::resources::Platform::hip) {
    return Op<resource::host_platform, resource::hip_platform>::exec(std::forward<Args>(args)...);
  }
  if (src_platform == camp::resources::Platform::hip && dst_platform == camp::resources::Platform::host) {
    return Op<resource::hip_platform, resource::host_platform>::exec(std::forward<Args>(args)...);
  }
#endif

#if defined(UMPIRE_ENABLE_SYCL)
  if (src_platform == camp::resources::Platform::host && dst_platform == camp::resources::Platform::sycl) {
    return Op<resource::host_platform, resource::sycl_platform>::exec(std::forward<Args>(args)...);
  }
  if (src_platform == camp::resources::Platform::sycl && dst_platform == camp::resources::Platform::host) {
    return Op<resource::sycl_platform, resource::host_platform>::exec(std::forward<Args>(args)...);
  }
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
  if (src_platform == camp::resources::Platform::host && dst_platform == camp::resources::Platform::omp_target) {
    return Op<resource::host_platform, resource::openmp_target_platform>::exec(std::forward<Args>(args)...);
  }
  if (src_platform == camp::resources::Platform::omp_target && dst_platform == camp::resources::Platform::host) {
    return Op<resource::openmp_target_platform, resource::host_platform>::exec(std::forward<Args>(args)...);
  }
#endif

  UMPIRE_ERROR(runtime_error, "Unsupported platform combination");
}

// Base template for op_caller with helper functions for argument handling
template <template <typename...> class Op>
struct op_caller {
  // Get the last argument from a parameter pack
  template <typename... Args>
  static auto get_last_arg(Args... args)
  {
    return std::get<sizeof...(Args) - 1>(std::forward_as_tuple(args...));
  }

  // Get the Nth argument from a parameter pack
  template <size_t N, typename... Args>
  static auto get_arg(Args... args)
  {
    return std::get<N>(std::forward_as_tuple(args...));
  }

  // Record operation event with common fields
  template <typename... Args>
  static void record_event(const char* name, void* ptr, void* strategy, const std::string& strategy_name,
                           bool async = false, Args&&... args)
  {
    umpire::event::record([&](auto& event) {
      event.name(name)
          .category(event::category::operation)
          .arg("ptr", ptr)
          .arg("allocator_ref", strategy)
          .tag("allocator_name", strategy_name)
          .tag("replay", "true");

      if (async) {
        event.tag("async", "true");
      }

      // Add any additional arguments
      add_args(event, std::forward<Args>(args)...);
    });
  }

  // Base case for argument processing
  template <typename Event>
  static void add_args(Event&)
  {
    // No more arguments to process
  }

  // Process key-value pairs
  template <typename Event, typename Key, typename Value, typename... Rest>
  static void add_args(Event& event, Key&& key, Value&& value, Rest&&... rest)
  {
    event.arg(std::forward<Key>(key), std::forward<Value>(value));
    add_args(event, std::forward<Rest>(rest)...);
  }

  // Boundary check for memset operations
  template <typename T, typename... Args>
  static void check_memset_bounds(T* src, const util::AllocationRecord* record, std::size_t length)
  {
    std::ptrdiff_t offset = static_cast<char*>(src) - static_cast<char*>(record->ptr);
    std::size_t size = record->size - offset;

    if (length > 0 && length > size) {
      UMPIRE_ERROR(runtime_error, fmt::format("Cannot memset over the end of allocation: {} -> {}", length, size));
    }
  }

  // Boundary check for copy operations
  template <typename T, typename... Args>
  static void check_copy_bounds(T* src, T* dst, const util::AllocationRecord* src_record,
                                const util::AllocationRecord* dst_record, std::size_t size)
  {
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
  }

  // Single-pointer operations (synchronous)
  template <typename T, typename... Args>
  inline static auto exec(T* src, Args... args)
  {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();

    // Operation-specific handling
    if constexpr (std::is_same_v<Op<resource::host_platform>, memset<resource::host_platform>>) {
      // For memset, we expect args to be {value, size}
      int value = get_arg<0>(args...);
      std::size_t length = get_arg<1>(args...);
      check_memset_bounds(src, src_record, length);
      record_event("memset", src, src_record->strategy, src_record->strategy->getName(), false, "value", value, "size",
                   length);
    } else if constexpr (std::is_same_v<Op<resource::host_platform>, prefetch<resource::host_platform>>) {
      // For prefetch, we expect args to be {device, size}
      int device = get_arg<0>(args...);
      std::size_t size = get_arg<1>(args...);
      record_event("prefetch", src, src_record->strategy, src_record->strategy->getName(), false, "device", device,
                   "size", size);
    }

    // Dispatch based on platform
    return dispatch_by_platform<Op>(p, src, args...);
  }

  // Single-pointer operations (asynchronous)
  template <typename T, typename... Args>
  inline static auto exec(T* src, camp::resources::Resource& ctx, Args... args)
  {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(src);
    auto p = src_record->strategy->getPlatform();

    // Operation-specific handling
    if constexpr (std::is_same_v<Op<resource::host_platform>, memset<resource::host_platform>>) {
      // For memset, we expect args to be {value, size}
      int value = get_arg<0>(args...);
      std::size_t length = get_arg<1>(args...);
      check_memset_bounds(src, src_record, length);
      record_event("memset", src, src_record->strategy, src_record->strategy->getName(), true, "value", value, "size",
                   length);
    } else if constexpr (std::is_same_v<Op<resource::host_platform>, prefetch<resource::host_platform>>) {
      // For prefetch, we expect args to be {device, size}
      int device = get_arg<0>(args...);
      std::size_t size = get_arg<1>(args...);
      record_event("prefetch", src, src_record->strategy, src_record->strategy->getName(), true, "device", device,
                   "size", size);
    }

    return dispatch_by_platform<Op>(p, src, args...);
  }

  // Dual-pointer operations (synchronous)
  template <typename T, typename... Args>
  inline static auto exec(T* src, T* dst, Args... args)
  {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(src);
    auto dst_record = allocation_map.find(dst);

    auto p1 = src_record->strategy->getPlatform();
    auto p2 = dst_record->strategy->getPlatform();

    // Operation-specific handling for copy
    if constexpr (std::is_same_v<Op<resource::host_platform, resource::host_platform>,
                                 copy<resource::host_platform, resource::host_platform>>) {
      // For copy, we expect args to be {size}
      std::size_t size = get_arg<0>(args...);
      check_copy_bounds(src, dst, src_record, dst_record, size);

      std::ptrdiff_t src_offset = static_cast<char*>(src) - static_cast<char*>(src_record->ptr);
      std::ptrdiff_t dst_offset = static_cast<char*>(dst) - static_cast<char*>(dst_record->ptr);

      record_event("copy", src, src_record->strategy, src_record->strategy->getName(), false, "dst", dst, "src_offset",
                   src_offset, "dst_offset", dst_offset, "size", size, "dst_allocator_ref", (void*)dst_record->strategy,
                   "src_allocator_name", src_record->strategy->getName(), "dst_allocator_name",
                   dst_record->strategy->getName());
    }

    // Dispatch based on source and destination platforms
    return dispatch_dual_platforms<Op>(p1, p2, src, dst, args...);
  }

  // Dual-pointer operations (asynchronous)
  template <typename T, typename... Args>
  inline static auto exec(T* src, T* dst, Args... args, camp::resources::Resource& ctx)
  {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(src);
    auto dst_record = allocation_map.find(dst);

    auto p1 = src_record->strategy->getPlatform();
    auto p2 = dst_record->strategy->getPlatform();

    // Operation-specific handling for copy
    if constexpr (std::is_same_v<Op<resource::host_platform, resource::host_platform>,
                                 copy<resource::host_platform, resource::host_platform>>) {
      // For copy, we expect args to be {size}
      std::size_t size = get_arg<0>(args...);
      check_copy_bounds(src, dst, src_record, dst_record, size);

      std::ptrdiff_t src_offset = static_cast<char*>(src) - static_cast<char*>(src_record->ptr);
      std::ptrdiff_t dst_offset = static_cast<char*>(dst) - static_cast<char*>(dst_record->ptr);

      record_event("copy", src, src_record->strategy, src_record->strategy->getName(), true, "dst", dst, "src_offset",
                   src_offset, "dst_offset", dst_offset, "size", size, "dst_allocator_ref", (void*)dst_record->strategy,
                   "src_allocator_name", src_record->strategy->getName(), "dst_allocator_name",
                   dst_record->strategy->getName());
    }

    return dispatch_dual_platforms<Op>(p1, p2, src, dst, args...);
  }
};

} // namespace op

// Global operation implementations that use the op_caller

template <typename T>
auto copy(T* src, T* dst, std::size_t len)
{
  op::op_caller<op::copy>::exec(src, dst, len);
}

template <typename T>
auto copy(T* src, T* dst, std::size_t len, camp::resources::Resource& ctx)
{
  return op::op_caller<op::copy>::exec(src, dst, len, ctx);
}

template <typename T, typename V>
void memset(T* src, V v, std::size_t len)
{
  op::op_caller<op::memset>::exec(src, v, len);
}

template <typename T>
camp::resources::EventProxy<camp::resources::Resource> memset(T* src, int v, std::size_t len,
                                                              camp::resources::Resource& ctx)
{
  return op::op_caller<op::memset>::exec(src, v, len, ctx);
}

// Non-void pointer reallocate
template <typename T, typename std::enable_if<!std::is_void<T>::value, int>::type = 0>
T* reallocate(T* src, std::size_t size)
{
  // Handle null pointer case
  if (src == nullptr) {
    auto& rm = ResourceManager::getInstance();
    Allocator allocator = rm.getDefaultAllocator();
    return static_cast<T*>(allocator.allocate(size * sizeof(T)));
  }

  auto& allocation_map = ResourceManager::getInstance().m_allocations;
  auto src_record = allocation_map.find(src);
  auto p = src_record->strategy->getPlatform();

  // Platform-specific dispatch based on the pointer's platform
  switch (p) {
    case camp::resources::Platform::host:
      return op::generic_reallocate<resource::host_platform>::exec(src, size);
#if defined(UMPIRE_ENABLE_CUDA)
    case camp::resources::Platform::cuda:
      return op::generic_reallocate<resource::cuda_platform>::exec(src, size);
#endif
#if defined(UMPIRE_ENABLE_HIP)
    case camp::resources::Platform::hip:
      return op::generic_reallocate<resource::hip_platform>::exec(src, size);
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    case camp::resources::Platform::sycl:
      return op::generic_reallocate<resource::sycl_platform>::exec(src, size);
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    case camp::resources::Platform::omp_target:
      return op::generic_reallocate<resource::openmp_target_platform>::exec(src, size);
#endif
    default:
      UMPIRE_ERROR(runtime_error, "Unknown platform for reallocate operation");
  }
}

// Void pointer reallocate
template <typename T, typename std::enable_if<std::is_void<T>::value, int>::type = 0>
void* reallocate(T* src, std::size_t size)
{
  // Handle null pointer case
  if (src == nullptr) {
    auto& rm = ResourceManager::getInstance();
    Allocator allocator = rm.getDefaultAllocator();
    return allocator.allocate(size);
  }

  auto& allocation_map = ResourceManager::getInstance().m_allocations;
  auto src_record = allocation_map.find(src);
  auto p = src_record->strategy->getPlatform();

  // Platform-specific dispatch based on the pointer's platform
  switch (p) {
    case camp::resources::Platform::host:
      return op::generic_reallocate<resource::host_platform>::exec(src, size);
#if defined(UMPIRE_ENABLE_CUDA)
    case camp::resources::Platform::cuda:
      return op::generic_reallocate<resource::cuda_platform>::exec(src, size);
#endif
#if defined(UMPIRE_ENABLE_HIP)
    case camp::resources::Platform::hip:
      return op::generic_reallocate<resource::hip_platform>::exec(src, size);
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    case camp::resources::Platform::sycl:
      return op::generic_reallocate<resource::sycl_platform>::exec(src, size);
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    case camp::resources::Platform::omp_target:
      return op::generic_reallocate<resource::openmp_target_platform>::exec(src, size);
#endif
    default:
      UMPIRE_ERROR(runtime_error, "Unknown platform for reallocate operation");
  }
}

// Async reallocate implementation
template <typename T>
camp::resources::EventProxy<camp::resources::Resource> reallocate(T* src, std::size_t size,
                                                                  camp::resources::Resource& ctx)
{
  // Handle null pointer case
  if (src == nullptr) {
    auto& rm = ResourceManager::getInstance();
    Allocator allocator = rm.getDefaultAllocator();
    allocator.allocate(size * sizeof(T));
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

  // Get platform and dispatch to appropriate implementation
  auto& allocation_map = ResourceManager::getInstance().m_allocations;
  auto src_record = allocation_map.find(src);
  auto p = src_record->strategy->getPlatform();

  // Use platform-specific dispatch with generic fallback
  switch (p) {
    case camp::resources::Platform::host:
      return op::generic_reallocate<resource::host_platform>::exec(src, size, ctx);
#if defined(UMPIRE_ENABLE_CUDA)
    case camp::resources::Platform::cuda:
      return op::generic_reallocate<resource::cuda_platform>::exec(src, size, ctx);
#endif
#if defined(UMPIRE_ENABLE_HIP)
    case camp::resources::Platform::hip:
      return op::generic_reallocate<resource::hip_platform>::exec(src, size, ctx);
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    case camp::resources::Platform::sycl:
      return op::generic_reallocate<resource::sycl_platform>::exec(src, size, ctx);
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    case camp::resources::Platform::omp_target:
      return op::generic_reallocate<resource::openmp_target_platform>::exec(src, size, ctx);
#endif
    default:
      return op::generic_reallocate<resource::undefined_platform>::exec(src, size, ctx);
  }
}

// Explicit specialization for void* async reallocate
template <>
camp::resources::EventProxy<camp::resources::Resource> reallocate(void* src, std::size_t size,
                                                                  camp::resources::Resource& ctx)
{
  // Handle null pointer case
  if (src == nullptr) {
    auto& rm = ResourceManager::getInstance();
    Allocator allocator = rm.getDefaultAllocator();
    allocator.allocate(size);
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

  // Get platform and dispatch to appropriate implementation
  auto& allocation_map = ResourceManager::getInstance().m_allocations;
  auto src_record = allocation_map.find(src);
  auto p = src_record->strategy->getPlatform();

  // Use platform-specific dispatch with generic fallback
  switch (p) {
    case camp::resources::Platform::host:
      return op::generic_reallocate<resource::host_platform>::exec(src, size, ctx);
#if defined(UMPIRE_ENABLE_CUDA)
    case camp::resources::Platform::cuda:
      return op::generic_reallocate<resource::cuda_platform>::exec(src, size, ctx);
#endif
#if defined(UMPIRE_ENABLE_HIP)
    case camp::resources::Platform::hip:
      return op::generic_reallocate<resource::hip_platform>::exec(src, size, ctx);
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    case camp::resources::Platform::sycl:
      return op::generic_reallocate<resource::sycl_platform>::exec(src, size, ctx);
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    case camp::resources::Platform::omp_target:
      return op::generic_reallocate<resource::openmp_target_platform>::exec(src, size, ctx);
#endif
    default:
      return op::generic_reallocate<resource::undefined_platform>::exec(src, size, ctx);
  }
}

template <typename T>
camp::resources::EventProxy<camp::resources::Resource> prefetch(T* ptr, int device, std::size_t size,
                                                                camp::resources::Resource& ctx)
{
  auto& rm = ResourceManager::getInstance();
  auto& allocation_map = rm.m_allocations;
  auto src_record = allocation_map.find(ptr);
  auto p = src_record->strategy->getPlatform();

  // Dispatch based on platform
  switch (p) {
    case camp::resources::Platform::host:
      return op::prefetch<resource::host_platform>::exec(ptr, device, size, ctx);
#if defined(UMPIRE_ENABLE_CUDA)
    case camp::resources::Platform::cuda:
      return op::prefetch<resource::cuda_platform>::exec(ptr, device, size, ctx);
#endif
#if defined(UMPIRE_ENABLE_HIP)
    case camp::resources::Platform::hip:
      return op::prefetch<resource::hip_platform>::exec(ptr, device, size, ctx);
#endif
#if defined(UMPIRE_ENABLE_SYCL)
    case camp::resources::Platform::sycl:
      return op::prefetch<resource::sycl_platform>::exec(ptr, device, size, ctx);
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    case camp::resources::Platform::omp_target:
      return op::prefetch<resource::openmp_target_platform>::exec(ptr, device, size, ctx);
#endif
    default:
      UMPIRE_ERROR(runtime_error, "Unknown platform for operation");
  }
}

} // namespace umpire
