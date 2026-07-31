//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_cuda_um_memory_HPP
#define UMPIRE_resource_cuda_um_memory_HPP

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
 * \brief Backend allocator wrapper for CUDA unified (managed) memory.
 *
 * Mirrors the v1 `alloc::CudaMallocManagedAllocator`: allocations are
 * satisfied with `cudaMallocManaged` and released with `cudaFree`. Unlike
 * `cuda_default_allocator`, no device is explicitly selected before the
 * call -- allocations land on the currently active CUDA device, matching v1
 * behavior.
 */
struct cuda_um_allocator {
  /*!
   * \brief Allocate CUDA managed memory with `cudaMallocManaged`.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to unified memory, accessible from host and device.
   *
   * \throws out_of_memory_error if the backend reports an allocation
   *         failure.
   * \throws runtime_error if `cudaMallocManaged` fails for any other reason.
   */
  char* allocate(std::size_t size) {
    void* ptr = nullptr;
    cudaError_t error = ::cudaMallocManaged(&ptr, size);
    if (error != cudaSuccess) {
      if (error == cudaErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error,
                     fmt::format("cudaMallocManaged({} bytes) failed with error: {}",
                                 size, cudaGetErrorString(error)));
      } else {
        UMPIRE_ERROR(runtime_error,
                     fmt::format("cudaMallocManaged({} bytes) failed with error: {}",
                                 size, cudaGetErrorString(error)));
      }
    }

    return static_cast<char*>(ptr);
  }

  /*!
   * \brief Deallocate managed memory with `cudaFree`.
   *
   * This function is `noexcept` to match allocator expectations; backend
   * failures are intentionally swallowed.
   */
  void deallocate(char* ptr, std::size_t /* size */) noexcept {
    cudaError_t error = ::cudaFree(ptr);
    (void)error; // Cannot throw in deallocate -- this must be noexcept
  }
};

/*!
 * \brief API v2 resource for CUDA unified (managed) memory allocations.
 *
 * The default singleton name is "UM", matching the v1
 * `CudaUnifiedMemoryResourceFactory` naming convention. As in v1, this
 * resource reports `cuda_platform` (not `host_platform`) even though the
 * memory it returns is host-accessible, because it is backend-owned CUDA
 * managed memory.
 *
 * \tparam Allocator Backend allocator wrapper.
 * \tparam Tracking Whether allocations are recorded in the v2 registry.
 */
template<
  typename Allocator = cuda_um_allocator,
  bool Tracking = true
>
class cuda_um_memory : public memory_resource<cuda_platform, Allocator, Tracking> {
private:
  using base = memory_resource<cuda_platform, Allocator, Tracking>;

  // Singleton instance
  static cuda_um_memory& instance() {
    static cuda_um_memory inst;
    return inst;
  }

  // Private constructor for singleton
  cuda_um_memory()
    : base("UM")
  {}

public:
  //! \brief Return the default singleton for CUDA unified memory.
  static cuda_um_memory& get() {
    return instance();
  }

  /*!
   * \brief Construct a named CUDA unified memory resource.
   *
   * \param name Human-readable resource name.
   * \param alloc Allocator instance used for raw allocations.
   */
  explicit cuda_um_memory(const std::string& name, Allocator alloc = Allocator())
    : base(name, std::move(alloc))
  {}

  /*!
   * \brief Allocate CUDA unified memory.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to unified memory, or `nullptr` for a zero-byte request.
   *
   * \throws out_of_memory_error if the allocation cannot be satisfied.
   */
  void* allocate(std::size_t size) override {
    if (size == 0) {
      return nullptr;  // Match cudaMallocManaged behavior
    }

    void* ptr = base::allocator_.allocate(size);

    if (!ptr) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("cuda_um_memory: allocation of {} bytes failed", size));
    }

    if constexpr (Tracking) {
      base::track_allocation(ptr, size);
    }

    return ptr;
  }

  /*!
   * \brief Deallocate unified memory previously returned by this resource.
   *
   * \param ptr Pointer to release. `nullptr` is a no-op.
   */
  void deallocate(void* ptr) override {
    if (!ptr) return;  // nullptr deallocation is safe no-op

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    // cuda_um_allocator::deallocate is noexcept
    base::allocator_.deallocate(static_cast<char*>(ptr), 0);
  }
};

//! \brief Tracking-enabled CUDA unified memory resource alias.
using default_cuda_um_memory = cuda_um_memory<cuda_um_allocator, true>;
//! \brief CUDA unified memory resource alias with tracking disabled.
using fast_cuda_um_memory = cuda_um_memory<cuda_um_allocator, false>;  // No tracking overhead

} // namespace resource
} // namespace umpire

#endif // UMPIRE_ENABLE_CUDA

#endif // UMPIRE_resource_cuda_um_memory_HPP
