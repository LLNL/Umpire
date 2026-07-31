//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_cuda_device_const_memory_HPP
#define UMPIRE_resource_cuda_device_const_memory_HPP

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_CUDA)

#include <cuda_runtime_api.h>

#include <mutex>
#include <string>
#include <vector>

#include "fmt/format.h"
#include "umpire/memory_resource.hpp"
#include "umpire/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

//! \brief Maximum total size, in bytes, of the fixed CUDA `__constant__` buffer.
inline constexpr std::size_t cuda_device_const_max_size = 64 * 1024;

namespace detail {

/*!
 * \brief Fixed-size CUDA device constant-memory buffer.
 *
 * Declared `inline` (C++17) so that every translation unit including this
 * header shares exactly one definition of the symbol, matching the ODR-safe
 * pattern needed for a header-only v2 resource. This mirrors the single
 * file-scope `__constant__` buffer used by the v1
 * `CudaConstantMemoryResource` (`src/umpire/resource/CudaConstantMemoryResource.cu`),
 * except the v1 version lived in exactly one translation unit by
 * construction (a single .cu source file) rather than a header.
 */
__constant__ inline char cuda_device_const_buffer[cuda_device_const_max_size];

} // namespace detail

/*!
 * \brief Unused allocator placeholder for `cuda_device_const_memory`.
 *
 * `memory_resource` requires an `Allocator` type to store and default
 * construct, but constant-memory allocation is a fixed offset-bump scheme
 * with no generic allocate/deallocate concept. This empty struct only
 * satisfies that interface requirement; its member functions are never
 * called by `cuda_device_const_memory`.
 */
struct cuda_device_const_allocator {
  //! \brief Unused; present only to satisfy the allocator wrapper interface.
  char* allocate(std::size_t) { return nullptr; }
  //! \brief Unused; present only to satisfy the allocator wrapper interface.
  void deallocate(char*, std::size_t) noexcept {}
};

/*!
 * \brief API v2 resource for CUDA `__constant__` device memory.
 *
 * Faithfully ports the v1 `CudaConstantMemoryResource` bump-offset
 * allocator: this resource owns a single fixed 64KB `__constant__` buffer
 * and hands out sequential, non-overlapping ranges from it. Because the
 * backing store is a single symbol, allocations must be released in
 * reverse (LIFO) order -- deallocating anything but the most recent live
 * allocation is an error, exactly as in v1.
 *
 * Unlike the other v2 backend resources, this resource does not use the
 * `Allocator` template parameter to perform raw allocation (there is no
 * generic allocator concept for a fixed offset-bump buffer); the parameter
 * is retained only so this class fits the same `memory_resource` base as
 * every other v2 resource.
 *
 * \tparam Allocator Unused backend allocator wrapper (kept for interface
 *         symmetry with other v2 resources).
 * \tparam Tracking Whether allocations are recorded in the v2 registry.
 */
template<
  typename Allocator = cuda_device_const_allocator,
  bool Tracking = true
>
class cuda_device_const_memory : public memory_resource<cuda_platform, Allocator, Tracking> {
private:
  using base = memory_resource<cuda_platform, Allocator, Tracking>;

  std::mutex mutex_;
  std::size_t offset_{0};
  void* ptr_{nullptr};
  bool initialized_{false};
  // Stack of live allocation sizes, in allocation order. Because the
  // backing buffer is a single bump-offset region, deallocation must occur
  // in reverse order; this stack lets deallocate() recover the size of the
  // most recent live allocation without depending on the shared v2
  // registry (which does not track size on the deallocate(ptr)-only path).
  std::vector<std::size_t> live_sizes_;

  // Singleton instance
  static cuda_device_const_memory& instance() {
    static cuda_device_const_memory inst;
    return inst;
  }

  // Private constructor for singleton
  cuda_device_const_memory()
    : base("DEVICE_CONST")
  {}

  void ensure_initialized() {
    if (initialized_) return;

    cudaError_t error = ::cudaGetSymbolAddress(&ptr_, detail::cuda_device_const_buffer);
    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("cudaGetSymbolAddress failed with error: {}",
                               cudaGetErrorString(error)));
    }

    initialized_ = true;
  }

public:
  //! \brief Return the default singleton for CUDA constant memory.
  static cuda_device_const_memory& get() {
    return instance();
  }

  /*!
   * \brief Construct a named CUDA constant-memory resource.
   *
   * All named instances share the same fixed 64KB `__constant__` buffer, so
   * separate instances are not independent allocation arenas; the name is
   * for diagnostics only, matching v1 (where each `CudaConstantMemoryResource`
   * instance also bound to the same file-scope buffer).
   *
   * \param name Human-readable resource name.
   */
  explicit cuda_device_const_memory(const std::string& name)
    : base(name)
  {}

  /*!
   * \brief Allocate from the fixed CUDA constant-memory buffer.
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer into the shared `__constant__` buffer.
   *
   * \throws runtime_error if the request would exceed the 64KB buffer, or
   *         if the backing symbol address cannot be resolved.
   */
  void* allocate(std::size_t bytes) override {
    std::lock_guard<std::mutex> lock{mutex_};

    ensure_initialized();

    char* ret{static_cast<char*>(ptr_) + offset_};
    std::size_t new_offset = offset_ + bytes;

    if (new_offset > cuda_device_const_max_size) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("Max total size of constant allocations is 64KB, current size is {} bytes",
                               offset_));
    }

    offset_ = new_offset;
    live_sizes_.push_back(bytes);

    if constexpr (Tracking) {
      base::track_allocation(ret, bytes);
    }

    return static_cast<void*>(ret);
  }

  /*!
   * \brief Deallocate from the fixed CUDA constant-memory buffer.
   *
   * \param ptr Pointer to release. Must be the most recently allocated live
   *        pointer (LIFO order), matching v1 semantics.
   *
   * \throws runtime_error if `ptr` is not the most recent live allocation.
   */
  void deallocate(void* ptr) override {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock{mutex_};

    if (live_sizes_.empty()) {
      UMPIRE_ERROR(runtime_error, "CudaConstantMemory deallocations must be in reverse order");
    }

    std::size_t size = live_sizes_.back();

    if ((static_cast<char*>(ptr_) + (offset_ - size)) == static_cast<char*>(ptr)) {
      offset_ -= size;
      live_sizes_.pop_back();
    } else {
      UMPIRE_ERROR(runtime_error, "CudaConstantMemory deallocations must be in reverse order");
    }

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }
  }
};

//! \brief Tracking-enabled CUDA constant memory resource alias.
using default_cuda_device_const_memory = cuda_device_const_memory<cuda_device_const_allocator, true>;
//! \brief CUDA constant memory resource alias with tracking disabled.
using fast_cuda_device_const_memory = cuda_device_const_memory<cuda_device_const_allocator, false>;

} // namespace resource
} // namespace umpire

#endif // UMPIRE_ENABLE_CUDA

#endif // UMPIRE_resource_cuda_device_const_memory_HPP
