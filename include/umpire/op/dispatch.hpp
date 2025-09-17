#pragma once

#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/platform.hpp"

namespace umpire {
namespace op {
namespace detail {

/**
 * @brief Dispatch an operation to the appropriate platform implementation
 *
 * @tparam Op The operation template to dispatch
 * @tparam Args Argument types for the operation
 * @param platform The platform to dispatch to
 * @param args Arguments for the operation
 * @return Result of the operation
 */
template <template <typename...> class Op, typename... Args>
inline auto dispatch(camp::resources::Platform platform, Args&&... args)
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

/**
 * @brief Dispatch an operation between two platforms (same or different)
 *
 * @tparam Op The operation template to dispatch
 * @tparam Args Argument types for the operation
 * @param src_platform The source platform
 * @param dst_platform The destination platform
 * @param args Arguments for the operation
 * @return Result of the operation
 */
template <template <typename...> class Op, typename... Args>
inline auto dispatch(camp::resources::Platform src_platform, camp::resources::Platform dst_platform, Args&&... args)
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

template <typename T>
constexpr auto decay_ptr(T* ptr)
{
  if constexpr (std::is_pointer_v<T>) {
    return *ptr;
  } else {
    return ptr;
  }
}
} // namespace detail

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

  // Boundary check for memset operations
  template <typename T, typename... Args>
  static void check_memset_bounds(T* src, const util::AllocationRecord* record, std::size_t length)
  {
    std::ptrdiff_t offset = reinterpret_cast<const char*>(src) - reinterpret_cast<const char*>(record->ptr);
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
    std::ptrdiff_t src_offset = reinterpret_cast<const char*>(src) - reinterpret_cast<const char*>(src_record->ptr);
    std::size_t src_size = src_record->size - src_offset;

    std::ptrdiff_t dst_offset = reinterpret_cast<const char*>(dst) - reinterpret_cast<const char*>(dst_record->ptr);
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
    auto src_record = allocation_map.find(detail::decay_ptr(src));
    auto p = src_record->strategy->getPlatform();

    // Operation-specific handling
    if constexpr (std::is_same_v<Op<resource::host_platform>, memset<resource::host_platform>>) {
      // For memset, we expect args to be {value, size}
      int value = get_arg<0>(args...);
      std::size_t length = get_arg<1>(args...);
      check_memset_bounds(src, src_record, length);
    }

    // Dispatch based on platform
    return detail::dispatch<Op>(p, src, args...);
  }

  // Single-pointer operations (asynchronous)
  template <typename T, typename... Args>
  inline static auto exec(T* src, camp::resources::Resource& ctx, Args... args)
  {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(detail::decay_ptr(src));
    auto p = src_record->strategy->getPlatform();

    // Operation-specific handling
    if constexpr (std::is_same_v<Op<resource::host_platform>, memset<resource::host_platform>>) {
      // For memset, we expect args to be {value, size}
      int value = get_arg<0>(args...);
      std::size_t length = get_arg<1>(args...);
      check_memset_bounds(src, src_record, length);
    }

    return detail::dispatch<Op>(p, src, args...);
  }

  // Dual-pointer operations (synchronous)
  template <typename T, typename... Args>
  inline static auto exec(T* src, T* dst, Args... args)
  {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(detail::decay_ptr(src));
    auto dst_record = allocation_map.find(dst);

    auto p1 = src_record->strategy->getPlatform();
    auto p2 = dst_record->strategy->getPlatform();

    // Operation-specific handling for copy
    if constexpr (std::is_same_v<Op<resource::host_platform, resource::host_platform>,
                                 copy<resource::host_platform, resource::host_platform>>) {
      // For copy, we expect args to be {size}
      std::size_t size = get_arg<0>(args...);
      check_copy_bounds(src, dst, src_record, dst_record, size);
    }

    // Dispatch based on source and destination platforms
    return detail::dispatch<Op>(p1, p2, src, dst, args...);
  }

  // Dual-pointer operations (asynchronous)
  template <typename T, typename... Args>
  inline static auto exec(T* src, T* dst, Args... args, camp::resources::Resource& ctx)
  {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(detail::decay_ptr(src));
    auto dst_record = allocation_map.find(dst);

    auto p1 = src_record->strategy->getPlatform();
    auto p2 = dst_record->strategy->getPlatform();

    // Operation-specific handling for copy
    if constexpr (std::is_same_v<Op<resource::host_platform, resource::host_platform>,
                                 copy<resource::host_platform, resource::host_platform>>) {
      // For copy, we expect args to be {size}
      std::size_t size = get_arg<0>(args...);
      check_copy_bounds(src, dst, src_record, dst_record, size);
    }

    return detail::dispatch<Op>(p1, p2, src, dst, args...);
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

template <typename T>
inline T* reallocate(T** src, std::size_t size)
{
  return op::op_caller<op::reallocate>::exec(src, size);
}

// Async reallocate implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> reallocate(T** src, std::size_t size,
                                                                         camp::resources::Resource& ctx)
{
  return op::op_caller<op::reallocate>::exec(src, size, ctx);
}

// Synchronous prefetch implementation
template <typename T>
void prefetch(T* ptr, int device, std::size_t size)
{
  op::op_caller<op::prefetch>::exec(ptr, device, size);
}

// Asynchronous prefetch implementation
template <typename T>
camp::resources::EventProxy<camp::resources::Resource> prefetch(T* ptr, int device, std::size_t size,
                                                                camp::resources::Resource& ctx)
{
  return op::op_caller<op::prefetch>::exec(ptr, device, size, ctx);
}

template <typename SrcPlatform, typename DstPlatform, typename T>
void copy(T* src, T* dst, std::size_t len)
{
  op::copy<SrcPlatform, DstPlatform>::exec(src, dst, len);
}

template <typename SrcPlatform, typename DstPlatform, typename T>
auto copy(T* src, T* dst, std::size_t len, camp::resources::Resource& ctx)
{
  return op::copy<SrcPlatform, DstPlatform>::exec(src, dst, len, ctx);
}

// Direct template memset functions
template <typename Platform, typename T>
void memset(T* ptr, int value, std::size_t len)
{
  op::memset<Platform>::exec(ptr, value, len);
}

template <typename Platform, typename T>
auto memset(T* ptr, int value, std::size_t len, camp::resources::Resource& ctx)
{
  return op::memset<Platform>::exec(ptr, value, len, ctx);
}

// Direct template prefetch functions
template <typename Platform, typename T>
void prefetch(T* ptr, int device, std::size_t len)
{
  op::prefetch<Platform>::exec(ptr, device, len);
}

template <typename Platform, typename T>
auto prefetch(T* ptr, int device, std::size_t len, camp::resources::Resource& ctx)
{
  return op::prefetch<Platform>::exec(ptr, device, len, ctx);
}

} // namespace umpire
