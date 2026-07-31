//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_hip_pinned_memory_HPP
#define UMPIRE_resource_hip_pinned_memory_HPP

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <string>

#include "fmt/format.h"
#include "umpire/memory_resource.hpp"
#include "umpire/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Backend allocator wrapper for HIP pinned (page-locked) host memory.
 *
 * Mirrors the v1 `alloc::HipPinnedAllocator`'s default (unknown-granularity)
 * path: allocations are satisfied with `hipHostMalloc` (no flags) and
 * released with `hipHostFree`. Coherence-granularity flag support
 * (`hipHostMallocDefault` / `hipHostMallocNonCoherent`) is not ported because
 * the v2 allocator wrapper interface has no traits/granularity plumbing yet.
 */
struct hip_pinned_allocator {
  /*!
   * \brief Allocate pinned host memory with `hipHostMalloc`.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to page-locked host memory.
   *
   * \throws out_of_memory_error if the backend reports an allocation
   *         failure.
   * \throws runtime_error if `hipHostMalloc` fails for any other reason.
   */
  char* allocate(std::size_t size) {
    void* ptr = nullptr;
    hipError_t error = ::hipHostMalloc(&ptr, size);
    if (error != hipSuccess) {
      if (error == hipErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error,
                     fmt::format("hipHostMalloc({} bytes) failed with error: {}",
                                 size, hipGetErrorString(error)));
      } else {
        UMPIRE_ERROR(runtime_error,
                     fmt::format("hipHostMalloc({} bytes) failed with error: {}",
                                 size, hipGetErrorString(error)));
      }
    }

    return static_cast<char*>(ptr);
  }

  /*!
   * \brief Deallocate pinned host memory with `hipHostFree`.
   *
   * This function is `noexcept` to match allocator expectations; backend
   * failures are intentionally swallowed.
   */
  void deallocate(char* ptr, std::size_t /* size */) noexcept {
    hipError_t error = ::hipHostFree(ptr);
    (void)error; // Cannot throw in deallocate -- this must be noexcept
  }
};

/*!
 * \brief API v2 resource for HIP pinned (page-locked) host memory
 *        allocations.
 *
 * The default singleton name is "PINNED", matching the v1
 * `HipPinnedMemoryResourceFactory` naming convention. As in v1, this
 * resource reports `hip_platform` (not `host_platform`) even though the
 * memory it returns is host-accessible, because it is backend-owned HIP
 * pinned memory.
 *
 * \tparam Allocator Backend allocator wrapper.
 * \tparam Tracking Whether allocations are recorded in the v2 registry.
 */
template<
  typename Allocator = hip_pinned_allocator,
  bool Tracking = true
>
class hip_pinned_memory : public memory_resource<hip_platform, Allocator, Tracking> {
private:
  using base = memory_resource<hip_platform, Allocator, Tracking>;

  // Singleton instance
  static hip_pinned_memory& instance() {
    static hip_pinned_memory inst;
    return inst;
  }

  // Private constructor for singleton
  hip_pinned_memory()
    : base("PINNED")
  {}

public:
  //! \brief Return the default singleton for HIP pinned memory.
  static hip_pinned_memory& get() {
    return instance();
  }

  /*!
   * \brief Construct a named HIP pinned memory resource.
   *
   * \param name Human-readable resource name.
   * \param alloc Allocator instance used for raw allocations.
   */
  explicit hip_pinned_memory(const std::string& name, Allocator alloc = Allocator())
    : base(name, std::move(alloc))
  {}

  /*!
   * \brief Allocate HIP pinned host memory.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to pinned memory, or `nullptr` for a zero-byte request.
   *
   * \throws out_of_memory_error if the allocation cannot be satisfied.
   */
  void* allocate(std::size_t size) override {
    if (size == 0) {
      return nullptr;  // Match hipHostMalloc behavior
    }

    void* ptr = base::allocator_.allocate(size);

    if (!ptr) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("hip_pinned_memory: allocation of {} bytes failed", size));
    }

    if constexpr (Tracking) {
      base::track_allocation(ptr, size);
    }

    return ptr;
  }

  /*!
   * \brief Deallocate pinned memory previously returned by this resource.
   *
   * \param ptr Pointer to release. `nullptr` is a no-op.
   */
  void deallocate(void* ptr) noexcept override {
    if (!ptr) return;  // nullptr deallocation is safe no-op

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    // hip_pinned_allocator::deallocate is noexcept
    base::allocator_.deallocate(static_cast<char*>(ptr), 0);
  }
};

//! \brief Tracking-enabled HIP pinned memory resource alias.
using default_hip_pinned_memory = hip_pinned_memory<hip_pinned_allocator, true>;
//! \brief HIP pinned memory resource alias with tracking disabled.
using fast_hip_pinned_memory = hip_pinned_memory<hip_pinned_allocator, false>;  // No tracking overhead

} // namespace resource
} // namespace umpire

#endif // UMPIRE_ENABLE_HIP

#endif // UMPIRE_resource_hip_pinned_memory_HPP
