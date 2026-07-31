//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_cuda_pinned_memory_HPP
#define UMPIRE_resource_cuda_pinned_memory_HPP

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_CUDA)

#include <cuda_runtime_api.h>
#include <string>

#include "fmt/format.h"
#include "umpire/memory_resource.hpp"
#include "umpire/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Backend allocator wrapper for CUDA pinned (page-locked) host memory.
 *
 * Mirrors the v1 `alloc::CudaPinnedAllocator`: allocations are satisfied
 * with `cudaMallocHost` and released with `cudaFreeHost`.
 */
struct cuda_pinned_allocator {
  /*!
   * \brief Allocate pinned host memory with `cudaMallocHost`.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to page-locked host memory.
   *
   * \throws out_of_memory_error if the backend reports an allocation
   *         failure.
   * \throws runtime_error if `cudaMallocHost` fails for any other reason.
   */
  char* allocate(std::size_t size) {
    void* ptr = nullptr;
    cudaError_t error = ::cudaMallocHost(&ptr, size);
    if (error != cudaSuccess) {
      if (error == cudaErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error,
                     fmt::format("cudaMallocHost({} bytes) failed with error: {}",
                                 size, cudaGetErrorString(error)));
      } else {
        UMPIRE_ERROR(runtime_error,
                     fmt::format("cudaMallocHost({} bytes) failed with error: {}",
                                 size, cudaGetErrorString(error)));
      }
    }

    return static_cast<char*>(ptr);
  }

  /*!
   * \brief Deallocate pinned host memory with `cudaFreeHost`.
   *
   * This function is `noexcept` to match allocator expectations; backend
   * failures are intentionally swallowed.
   */
  void deallocate(char* ptr, std::size_t /* size */) noexcept {
    cudaError_t error = ::cudaFreeHost(ptr);
    (void)error; // Cannot throw in deallocate -- this must be noexcept
  }
};

/*!
 * \brief API v2 resource for CUDA pinned (page-locked) host memory
 *        allocations.
 *
 * The default singleton name is "PINNED", matching the v1
 * `CudaPinnedMemoryResourceFactory` naming convention. As in v1, this
 * resource reports `cuda_platform` (not `host_platform`) even though the
 * memory it returns is host-accessible, because it is backend-owned CUDA
 * pinned memory.
 *
 * \tparam Allocator Backend allocator wrapper.
 * \tparam Tracking Whether allocations are recorded in the v2 registry.
 */
template<
  typename Allocator = cuda_pinned_allocator,
  bool Tracking = true
>
class cuda_pinned_memory : public memory_resource<cuda_platform, Allocator, Tracking> {
private:
  using base = memory_resource<cuda_platform, Allocator, Tracking>;

  // Singleton instance
  static cuda_pinned_memory& instance() {
    static cuda_pinned_memory inst;
    return inst;
  }

  // Private constructor for singleton
  cuda_pinned_memory()
    : base("PINNED")
  {}

public:
  //! \brief Return the default singleton for CUDA pinned memory.
  static cuda_pinned_memory& get() {
    return instance();
  }

  /*!
   * \brief Construct a named CUDA pinned memory resource.
   *
   * \param name Human-readable resource name.
   * \param alloc Allocator instance used for raw allocations.
   */
  explicit cuda_pinned_memory(const std::string& name, Allocator alloc = Allocator())
    : base(name, std::move(alloc))
  {}

  /*!
   * \brief Allocate CUDA pinned host memory.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to pinned memory, or `nullptr` for a zero-byte request.
   *
   * \throws out_of_memory_error if the allocation cannot be satisfied.
   */
  void* allocate(std::size_t size) override {
    if (size == 0) {
      return nullptr;  // Match cudaMallocHost behavior
    }

    void* ptr = base::allocator_.allocate(size);

    if (!ptr) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("cuda_pinned_memory: allocation of {} bytes failed", size));
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
  void deallocate(void* ptr) override {
    if (!ptr) return;  // nullptr deallocation is safe no-op

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    // cuda_pinned_allocator::deallocate is noexcept
    base::allocator_.deallocate(static_cast<char*>(ptr), 0);
  }
};

//! \brief Tracking-enabled CUDA pinned memory resource alias.
using default_cuda_pinned_memory = cuda_pinned_memory<cuda_pinned_allocator, true>;
//! \brief CUDA pinned memory resource alias with tracking disabled.
using fast_cuda_pinned_memory = cuda_pinned_memory<cuda_pinned_allocator, false>;  // No tracking overhead

} // namespace resource
} // namespace umpire

#endif // UMPIRE_ENABLE_CUDA

#endif // UMPIRE_resource_cuda_pinned_memory_HPP
