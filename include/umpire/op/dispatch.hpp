#pragma once

#include <type_traits>

#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/memory.hpp"
#include "umpire/op/detail/traits.hpp"
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

inline auto find_v2_allocation(void* ptr)
{
  auto& registry = ::umpire::detail::registry::get();
  auto record = registry.find_allocation(ptr);
  if (record && record->strategy) {
    return record;
  }

  record = registry.find_containing_allocation(ptr);
  if (record && record->strategy) {
    if (ptr != record->ptr) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("Cannot reallocate an offset ptr (ptr={}, base={})", ptr, record->ptr));
    }
    return record;
  }

  return std::optional<::umpire::allocation_record>{};
}

template <typename T>
inline std::size_t reallocate_size_bytes(std::size_t new_size)
{
  if constexpr (std::is_void_v<T>) {
    return new_size;
  } else {
    return new_size * sizeof(T);
  }
}

template <typename T>
inline T* reallocate_v2(T** ptr, std::size_t new_size)
{
  auto* current_ptr = *ptr;
  auto record = find_v2_allocation(current_ptr);

  if (!record) {
    return nullptr;
  }

  auto* owner = record->strategy;
  const auto platform = owner->get_platform();
  const std::size_t old_bytes = record->size;
  const std::size_t new_bytes = reallocate_size_bytes<T>(new_size);

  if (new_bytes == 0) {
    owner->deallocate(current_ptr);
    auto* new_ptr = static_cast<T*>(owner->allocate(0));
    *ptr = new_ptr;
    return new_ptr;
  }

  auto* new_ptr = static_cast<T*>(owner->allocate(new_bytes));
  const std::size_t copy_bytes = min(old_bytes, new_bytes);

  dispatch<copy>(platform, platform, static_cast<void*>(current_ptr), static_cast<void*>(new_ptr), copy_bytes);
  owner->deallocate(current_ptr);
  *ptr = new_ptr;

  return new_ptr;
}

template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> reallocate_v2_async(T** ptr, std::size_t new_size,
                                                                                   camp::resources::Resource& ctx)
{
  auto* current_ptr = *ptr;
  auto record = find_v2_allocation(current_ptr);

  if (!record) {
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

  auto* owner = record->strategy;
  const auto platform = owner->get_platform();
  const std::size_t old_bytes = record->size;
  const std::size_t new_bytes = reallocate_size_bytes<T>(new_size);

  if (new_bytes == 0) {
    owner->deallocate(current_ptr);
    auto* new_ptr = static_cast<T*>(owner->allocate(0));
    *ptr = new_ptr;
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

  auto* new_ptr = static_cast<T*>(owner->allocate(new_bytes));
  const std::size_t copy_bytes = min(old_bytes, new_bytes);
  auto event = dispatch<copy>(
      platform, platform, static_cast<void*>(current_ptr), static_cast<void*>(new_ptr), copy_bytes, ctx);

  // Without chained events, wait before deallocating to avoid freeing in-flight source storage.
  ctx.get_event().wait();
  owner->deallocate(current_ptr);
  *ptr = new_ptr;

  return event;
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

#ifdef UMPIRE_ENABLE_BOUNDS_CHECKS
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
#endif // UMPIRE_ENABLE_BOUNDS_CHECKS

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
#ifdef UMPIRE_ENABLE_BOUNDS_CHECKS
      std::size_t length = get_arg<1>(args...);
      check_memset_bounds(src, src_record, length);
#endif
    }

    // Dispatch based on platform
    return detail::dispatch<Op>(p, src, args...);
  }

  // Single-pointer operations (asynchronous)
  template <typename T, typename... Args>
  inline static auto exec(T* src, Args... args, camp::resources::Resource& ctx)
  {
    auto& rm = ResourceManager::getInstance();
    auto& allocation_map = rm.m_allocations;
    auto src_record = allocation_map.find(detail::decay_ptr(src));
    auto p = src_record->strategy->getPlatform();

    // Operation-specific handling
    if constexpr (std::is_same_v<Op<resource::host_platform>, memset<resource::host_platform>>) {
      // For memset, we expect args to be {value, size}
#ifdef UMPIRE_ENABLE_BOUNDS_CHECKS
      std::size_t length = get_arg<1>(args...);
      check_memset_bounds(src, src_record, length);
#endif
    }

    return detail::dispatch<Op>(p, src, std::forward<Args>(args)..., ctx);
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
#ifdef UMPIRE_ENABLE_BOUNDS_CHECKS
      std::size_t size = get_arg<0>(args...);
      check_copy_bounds(src, dst, src_record, dst_record, size);
#endif
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
#ifdef UMPIRE_ENABLE_BOUNDS_CHECKS
      std::size_t size = get_arg<0>(args...);
      check_copy_bounds(src, dst, src_record, dst_record, size);
#endif
    }

    return detail::dispatch<Op>(p1, p2, src, dst, std::forward<Args>(args)..., ctx);
  }
};

} // namespace op

//------------------------------------------------------------------------------
// Public API Semantics for Copy and Reallocate Operations
//------------------------------------------------------------------------------
//
// The following functions use different semantics for size parameters based on
// pointer type, matching standard C++ conventions:
//
// COPY OPERATIONS:
//   - umpire::copy(T* src, T* dst, std::size_t len) for non-void T:
//     len is a COUNT OF ELEMENTS (will be multiplied by sizeof(T) internally)
//
//   - umpire::copy(void* src, void* dst, std::size_t len):
//     len is BYTES (used directly, no sizeof multiplication)
//
// REALLOCATE OPERATIONS:
//   - umpire::reallocate(T** ptr, std::size_t new_size) for non-void T:
//     new_size is a COUNT OF ELEMENTS (will be multiplied by sizeof(T) internally)
//
//   - umpire::reallocate(void** ptr, std::size_t new_size):
//     new_size is BYTES (used directly, no sizeof multiplication)
//
// RATIONALE:
//   This matches how detail::get_size<T>(count) works:
//   - For void: returns count as-is (bytes)
//   - For typed pointers: returns count * sizeof(T) (elements to bytes)
//
// This keeps the API consistent with C++ idioms where typed operations work
// with element counts and void* operations work with byte counts.
//------------------------------------------------------------------------------

// Global operation implementations that use the op_caller
template <typename T>
void copy(T* src, T* dst, std::size_t len)
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
  // Handle nullptr specially - op_caller can't look up null in allocation map
  if (*src == nullptr) {
    return op::reallocate<resource::host_platform>::exec(src, size);
  }

  if (auto record = op::detail::find_v2_allocation(*src)) {
    return op::detail::dispatch<op::reallocate>(record->strategy->get_platform(), src, size);
  }

  return op::op_caller<op::reallocate>::exec(src, size);
}

// Async reallocate implementation
template <typename T>
inline camp::resources::EventProxy<camp::resources::Resource> reallocate(T** src, std::size_t size,
                                                                         camp::resources::Resource& ctx)
{
  // Handle nullptr specially - op_caller can't look up null in allocation map
  if (*src == nullptr) {
    return op::reallocate<resource::host_platform>::exec(src, size, ctx);
  }

  if (auto record = op::detail::find_v2_allocation(*src)) {
    return op::detail::dispatch<op::reallocate>(record->strategy->get_platform(), src, size, ctx);
  }

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

// Individual convenience functions with auto-dispatch
template <typename T>
void set_accessed_by(T* ptr, int device, std::size_t size)
{
  op::op_caller<op::set_accessed_by>::exec(ptr, device, size);
}

template <typename T>
void set_preferred_location(T* ptr, int device, std::size_t size)
{
  op::op_caller<op::set_preferred_location>::exec(ptr, device, size);
}

template <typename T>
void set_read_mostly(T* ptr, int device, std::size_t size)
{
  op::op_caller<op::set_read_mostly>::exec(ptr, device, size);
}

template <typename T>
void unset_accessed_by(T* ptr, int device, std::size_t size)
{
  op::op_caller<op::unset_accessed_by>::exec(ptr, device, size);
}

template <typename T>
void unset_preferred_location(T* ptr, int device, std::size_t size)
{
  op::op_caller<op::unset_preferred_location>::exec(ptr, device, size);
}

template <typename T>
void unset_read_mostly(T* ptr, int device, std::size_t size)
{
  op::op_caller<op::unset_read_mostly>::exec(ptr, device, size);
}

#if (defined(UMPIRE_ENABLE_HIP) && HIP_VERSION_MAJOR >= 5)
template <typename T>
void set_coarse_grain(T* ptr, int device, std::size_t size)
{
  op::op_caller<op::set_coarse_grain>::exec(ptr, device, size);
}

template <typename T>
void unset_coarse_grain(T* ptr, int device, std::size_t size)
{
  op::op_caller<op::unset_coarse_grain>::exec(ptr, device, size);
}
#endif

//------------------------------------------------------------------------------
// Reallocate implementations (moved from operations.hpp to avoid circular dependency)
//------------------------------------------------------------------------------

template <typename Src>
template <typename T>
T* op::reallocate<Src>::exec(T** ptr, std::size_t new_size)
{
  auto current_ptr = *ptr;
  if (!current_ptr) {
    // If current pointer is null, just allocate
    auto& rm = ResourceManager::getInstance();
    Allocator allocator = rm.getDefaultAllocator();
    T* new_ptr = static_cast<T*>(allocator.allocate(new_size * sizeof(T)));
    *ptr = new_ptr;
    return new_ptr;
  }

  if (auto* new_ptr = detail::reallocate_v2(ptr, new_size)) {
    return new_ptr;
  }

  auto& rm = ResourceManager::getInstance();
  auto& allocation_map = rm.m_allocations;

  // Find the allocator that owns current_ptr
  Allocator allocator = rm.getAllocator(current_ptr);

  // Check for offset pointer
  auto alloc_record = allocation_map.find(current_ptr);
  if (current_ptr != alloc_record->ptr) {
    UMPIRE_ERROR(runtime_error, fmt::format("Cannot reallocate an offset ptr (ptr={}, base={})",
                                            reinterpret_cast<void*>(current_ptr), alloc_record->ptr));
  }

  // Get the current allocation size
  std::size_t old_size = rm.getSize(current_ptr);

  // Convert sizes from elements to bytes
  std::size_t old_bytes = old_size;
  std::size_t new_bytes = new_size * sizeof(T);

  // Special case for 0-byte size
  if (new_bytes == 0) {
    allocator.deallocate(current_ptr);
    T* new_ptr = static_cast<T*>(allocator.allocate(0));
    *ptr = new_ptr;
    return new_ptr;
  }

  // Allocate new memory
  T* new_ptr = static_cast<T*>(allocator.allocate(new_bytes));

  // Calculate copy size in bytes (minimum of old and new size)
  std::size_t copy_bytes = (old_bytes > new_bytes) ? new_bytes : old_bytes;

  // Copy data using void* to pass bytes directly (avoids sizeof(T) multiplication in copy)
  // Note: We cast to void* so that detail::get_size<void>(len) returns len as-is (bytes)
  umpire::copy(static_cast<void*>(current_ptr), static_cast<void*>(new_ptr), copy_bytes);

  // Deallocate old memory
  allocator.deallocate(current_ptr);

  // Update the pointer
  *ptr = new_ptr;

  return new_ptr;
}

template <typename Src>
template <typename T>
camp::resources::EventProxy<camp::resources::Resource> op::reallocate<Src>::exec(T** ptr_ptr, std::size_t new_size,
                                                                                 camp::resources::Resource& ctx)
{
  T* current_ptr = *ptr_ptr;

  if (!current_ptr) {
    // If current pointer is null, just allocate
    auto& rm = ResourceManager::getInstance();
    Allocator allocator = rm.getDefaultAllocator();
    // Since there's no data to copy, we can just return a completed event
    T* new_ptr = static_cast<T*>(allocator.allocate(new_size * sizeof(T)));
    *ptr_ptr = new_ptr;
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

  if (detail::find_v2_allocation(current_ptr)) {
    return detail::reallocate_v2_async(ptr_ptr, new_size, ctx);
  }

  auto& rm = ResourceManager::getInstance();
  auto& allocation_map = rm.m_allocations;

  // Find the allocator that owns current_ptr
  Allocator allocator = rm.getAllocator(current_ptr);

  // Check for offset pointer
  auto alloc_record = allocation_map.find(current_ptr);
  if (current_ptr != alloc_record->ptr) {
    UMPIRE_ERROR(runtime_error, fmt::format("Cannot reallocate an offset ptr (ptr={}, base={})",
                                            reinterpret_cast<void*>(current_ptr), alloc_record->ptr));
  }

  // Get the current allocation size
  std::size_t old_size = rm.getSize(current_ptr);

  // Convert sizes from elements to bytes
  std::size_t old_bytes = old_size;
  std::size_t new_bytes = new_size * sizeof(T);

  // Special case for 0-byte size
  if (new_bytes == 0) {
    allocator.deallocate(current_ptr);
    T* new_ptr = static_cast<T*>(allocator.allocate(0));
    *ptr_ptr = new_ptr;
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

  // Allocate new memory
  T* new_ptr = static_cast<T*>(allocator.allocate(new_bytes));

  // Calculate copy size in bytes (minimum of old and new size)
  std::size_t copy_bytes = (old_bytes > new_bytes) ? new_bytes : old_bytes;

  // Copy data using void* to pass bytes directly (avoids sizeof(T) multiplication in copy)
  // Note: We cast to void* so that detail::get_size<void>(len) returns len as-is (bytes)
  auto event = umpire::copy(static_cast<void*>(current_ptr), static_cast<void*>(new_ptr), copy_bytes, ctx);

  // Without chained events, wait before deallocating to avoid freeing in-flight source storage.
  ctx.get_event().wait();
  allocator.deallocate(current_ptr);

  // Update the pointer
  *ptr_ptr = new_ptr;

  return event;
}

template <typename Src>
void* op::reallocate<Src>::exec(void** ptr_ptr, std::size_t new_size)
{
  void* current_ptr = *ptr_ptr;

  if (!current_ptr) {
    // If current pointer is null, just allocate
    auto& rm = ResourceManager::getInstance();
    Allocator allocator = rm.getDefaultAllocator();
    void* new_ptr = allocator.allocate(new_size); // No sizeof multiplication for void*
    *ptr_ptr = new_ptr;
    return new_ptr;
  }

  if (auto* new_ptr = detail::reallocate_v2(ptr_ptr, new_size)) {
    return new_ptr;
  }

  auto& rm = ResourceManager::getInstance();
  auto& allocation_map = rm.m_allocations;

  // Find the allocator that owns current_ptr
  Allocator allocator = rm.getAllocator(current_ptr);

  // Check for offset pointer
  auto alloc_record = allocation_map.find(current_ptr);
  if (current_ptr != alloc_record->ptr) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("Cannot reallocate an offset ptr (ptr={}, base={})", current_ptr, alloc_record->ptr));
  }

  // Get the current allocation size
  std::size_t old_size = rm.getSize(current_ptr);

  // Special case for 0-byte size
  if (new_size == 0) {
    allocator.deallocate(current_ptr);
    void* new_ptr = allocator.allocate(0);
    *ptr_ptr = new_ptr;
    return new_ptr;
  }

  // Allocate new memory
  void* new_ptr = allocator.allocate(new_size);

  // Calculate copy size in bytes (minimum of old and new size)
  std::size_t copy_bytes = (old_size > new_size) ? new_size : old_size;

  // Copy data from old to new location (void* naturally uses bytes)
  umpire::copy(static_cast<void*>(current_ptr), static_cast<void*>(new_ptr), copy_bytes);

  // Deallocate old memory
  allocator.deallocate(current_ptr);

  // Update the pointer
  *ptr_ptr = new_ptr;

  return new_ptr;
}

template <typename Src>
camp::resources::EventProxy<camp::resources::Resource> op::reallocate<Src>::exec(void** ptr_ptr, std::size_t new_size,
                                                                                 camp::resources::Resource& ctx)
{
  void* current_ptr = *ptr_ptr;

  if (!current_ptr) {
    // If current pointer is null, just allocate
    auto& rm = ResourceManager::getInstance();
    Allocator allocator = rm.getDefaultAllocator();
    // Since there's no data to copy, we can just return a completed event
    void* new_ptr = allocator.allocate(new_size); // No sizeof multiplication for void*
    *ptr_ptr = new_ptr;
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

  if (detail::find_v2_allocation(current_ptr)) {
    return detail::reallocate_v2_async(ptr_ptr, new_size, ctx);
  }

  auto& rm = ResourceManager::getInstance();
  auto& allocation_map = rm.m_allocations;

  // Find the allocator that owns current_ptr
  Allocator allocator = rm.getAllocator(current_ptr);

  // Check for offset pointer
  auto alloc_record = allocation_map.find(current_ptr);
  if (current_ptr != alloc_record->ptr) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("Cannot reallocate an offset ptr (ptr={}, base={})", current_ptr, alloc_record->ptr));
  }

  // Get the current allocation size
  std::size_t old_size = rm.getSize(current_ptr);

  // Special case for 0-byte size
  if (new_size == 0) {
    allocator.deallocate(current_ptr);
    void* new_ptr = allocator.allocate(0);
    *ptr_ptr = new_ptr;
    return camp::resources::EventProxy<camp::resources::Resource>{ctx};
  }

  // Allocate new memory
  void* new_ptr = allocator.allocate(new_size);

  // Calculate copy size in bytes (minimum of old and new size)
  std::size_t copy_bytes = (old_size > new_size) ? new_size : old_size;

  // Copy data from old to new location asynchronously (void* naturally uses bytes)
  auto event = umpire::copy(static_cast<void*>(current_ptr), static_cast<void*>(new_ptr), copy_bytes, ctx);

  // Wait for the async copy to complete before freeing the source allocation.
  ctx.get_event().wait();
  allocator.deallocate(current_ptr);

  // Update the pointer
  *ptr_ptr = new_ptr;

  return event;
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

template <typename Platform, typename T>
std::enable_if_t<op::detail::supports_memory_advice<Platform>::value> set_accessed_by(T* ptr, int device,
                                                                                      std::size_t len)
{
  op::set_accessed_by<Platform>::exec(ptr, device, len);
}

template <typename Platform, typename T>
std::enable_if_t<op::detail::supports_memory_advice<Platform>::value> set_preferred_location(T* ptr, int device,
                                                                                             std::size_t len)
{
  op::set_preferred_location<Platform>::exec(ptr, device, len);
}

template <typename Platform, typename T>
std::enable_if_t<op::detail::supports_memory_advice<Platform>::value> set_read_mostly(T* ptr, int device,
                                                                                      std::size_t len)
{
  op::set_read_mostly<Platform>::exec(ptr, device, len);
}

template <typename Platform, typename T>
std::enable_if_t<op::detail::supports_memory_advice<Platform>::value> unset_accessed_by(T* ptr, int device,
                                                                                        std::size_t len)
{
  op::unset_accessed_by<Platform>::exec(ptr, device, len);
}

template <typename Platform, typename T>
std::enable_if_t<op::detail::supports_memory_advice<Platform>::value> unset_preferred_location(T* ptr, int device,
                                                                                               std::size_t len)
{
  op::unset_preferred_location<Platform>::exec(ptr, device, len);
}

template <typename Platform, typename T>
std::enable_if_t<op::detail::supports_memory_advice<Platform>::value> unset_read_mostly(T* ptr, int device,
                                                                                        std::size_t len)
{
  op::unset_read_mostly<Platform>::exec(ptr, device, len);
}

#if (defined(UMPIRE_ENABLE_HIP) && HIP_VERSION_MAJOR >= 5) || defined(UMPIRE_ENABLE_CUDA)
template <typename Platform, typename T>
std::enable_if_t<op::detail::supports_memory_advice<Platform>::value> set_coarse_grain(T* ptr, int device,
                                                                                       std::size_t len)
{
  op::set_coarse_grain<Platform>::exec(ptr, device, len);
}

template <typename Platform, typename T>
std::enable_if_t<op::detail::supports_memory_advice<Platform>::value> unset_coarse_grain(T* ptr, int device,
                                                                                         std::size_t len)
{
  op::unset_coarse_grain<Platform>::exec(ptr, device, len);
}
#endif

} // namespace umpire
