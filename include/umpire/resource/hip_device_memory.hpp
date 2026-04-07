//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_hip_device_memory_HPP
#define UMPIRE_resource_hip_device_memory_HPP

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
 * \brief Backend allocator wrapper for HIP device memory.
 *
 * The allocator binds allocations to a specific HIP device and satisfies the
 * allocator interface expected by `memory_resource`.
 */
struct hip_default_allocator {
  int device_id;

  //! \brief Construct an allocator targeting `device`.
  explicit hip_default_allocator(int device = 0) : device_id(device) {}

  //! \brief Copy this allocator wrapper, preserving the targeted device.
  hip_default_allocator(const hip_default_allocator&) = default;
  //! \brief Assign from another allocator wrapper targeting a HIP device.
  hip_default_allocator& operator=(const hip_default_allocator&) = default;

  /*!
   * \brief Allocate device memory with `hipMalloc`.
   *
   * \param size Number of bytes to allocate on the configured device.
   * \return Pointer to device memory.
   *
   * \throws runtime_error if device selection fails.
   * \throws out_of_memory_error if `hipMalloc` fails.
   */
  char* allocate(std::size_t size) {
    // Set the device before allocation
    hipError_t error = ::hipSetDevice(device_id);
    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("hipSetDevice({}) failed with error: {}",
                               device_id, hipGetErrorString(error)));
    }

    void* ptr = nullptr;
    error = ::hipMalloc(&ptr, size);
    if (error != hipSuccess) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("hipMalloc({} bytes) on device {} failed with error: {}",
                               size, device_id, hipGetErrorString(error)));
    }

    return static_cast<char*>(ptr);
  }

  /*!
   * \brief Deallocate device memory with `hipFree`.
   *
   * This function is `noexcept` to match allocator expectations; backend
   * failures are intentionally swallowed.
   */
  void deallocate(char* ptr, std::size_t /* size */) noexcept {
    // hipFree doesn't require size parameter
    // Set device before deallocation
    hipError_t error = ::hipSetDevice(device_id);
    if (error != hipSuccess) {
      // Cannot throw in deallocate, but we should handle this gracefully
      // In production, this would be logged
      return;
    }

    error = ::hipFree(ptr);
    // Cannot throw in deallocate - this must be noexcept
    // In production, errors would be logged but not thrown
  }
};

/*!
 * \brief API v2 resource for HIP device allocations.
 *
 * The default singleton targets HIP device 0. Additional instances can bind to
 * other devices and participate in strategy composition.
 *
 * \tparam Allocator Backend allocator wrapper.
 * \tparam Tracking Whether allocations are recorded in the v2 registry.
 */
template<
  typename Allocator = hip_default_allocator,
  bool Tracking = true
>
class hip_device_memory : public memory_resource<hip_platform, Allocator, Tracking> {
private:
  using base = memory_resource<hip_platform, Allocator, Tracking>;

  int device_id_;

  // Singleton instance (for default config only - device 0)
  static hip_device_memory& instance() {
    static hip_device_memory inst;
    return inst;
  }

  // Private constructor for singleton (device 0)
  hip_device_memory()
    : base("HIP_DEVICE", hip_default_allocator(0))
    , device_id_(0)
  {}

public:
  //! \brief Return the default singleton for HIP device 0.
  static hip_device_memory& get() {
    return instance();
  }

  /*!
   * \brief Construct a named HIP resource for a specific device.
   *
   * \param name Human-readable resource name.
   * \param device_id HIP device ordinal to target.
   * \param alloc Allocator instance used for raw allocations.
   *
   * \throws runtime_error if the requested device is unavailable.
   */
  explicit hip_device_memory(const std::string& name, int device_id = 0,
                              Allocator alloc = Allocator())
    : base(name, std::move(alloc))
    , device_id_(device_id)
  {
    // Verify device exists and is valid
    int device_count = 0;
    hipError_t error = ::hipGetDeviceCount(&device_count);
    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("hipGetDeviceCount() failed with error: {}",
                               hipGetErrorString(error)));
    }

    if (device_id < 0 || device_id >= device_count) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("Invalid HIP device ID: {} (available devices: 0-{})",
                               device_id, device_count - 1));
    }
  }

  /*!
   * \brief Construct a HIP resource named from its device ordinal.
   *
   * \param device_id HIP device ordinal to target.
   */
  explicit hip_device_memory(int device_id)
    : hip_device_memory(fmt::format("HIP_DEVICE_{}", device_id), device_id,
                        hip_default_allocator(device_id))
  {}

  //! \brief Return the HIP device ordinal targeted by this resource.
  int get_device_id() const {
    return device_id_;
  }

  /*!
   * \brief Allocate HIP device memory.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to device memory, or `nullptr` for a zero-byte request.
   *
   * \throws out_of_memory_error if the allocation cannot be satisfied.
   */
  void* allocate(std::size_t size) override {
    if (size == 0) {
      return nullptr;  // Match hipMalloc behavior
    }

    void* ptr = base::allocator_.allocate(size);

    if (!ptr) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("hip_device_memory: allocation of {} bytes on device {} failed",
                               size, device_id_));
    }

    if constexpr (Tracking) {
      base::track_allocation(ptr, size);
    }

    return ptr;
  }

  /*!
   * \brief Deallocate device memory previously returned by this resource.
   *
   * \param ptr Pointer to release. `nullptr` is a no-op.
   */
  void deallocate(void* ptr) noexcept override {
    if (!ptr) return;  // nullptr deallocation is safe no-op

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    // hip_default_allocator::deallocate is noexcept
    base::allocator_.deallocate(static_cast<char*>(ptr), 0);
  }
};

//! \brief Tracking-enabled HIP device resource alias.
using default_hip_device_memory = hip_device_memory<hip_default_allocator, true>;
//! \brief HIP device resource alias with tracking disabled.
using fast_hip_device_memory = hip_device_memory<hip_default_allocator, false>;  // No tracking overhead

} // namespace resource
} // namespace umpire

//! \brief Backward-compatible alias for the default HIP allocator wrapper.
namespace umpire {
using hip_allocator = resource::hip_default_allocator;
}

#endif // UMPIRE_ENABLE_HIP

#endif // UMPIRE_resource_hip_device_memory_HPP
