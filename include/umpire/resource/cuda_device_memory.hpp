//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_cuda_device_memory_HPP
#define UMPIRE_resource_cuda_device_memory_HPP

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

// CUDA allocator wrapper that uses cudaMalloc/cudaFree
// Doesn't require size in deallocate (cudaFree doesn't need it)
struct cuda_default_allocator {
  int device_id;

  explicit cuda_default_allocator(int device = 0) : device_id(device) {}

  // Copy constructor
  cuda_default_allocator(const cuda_default_allocator&) = default;
  cuda_default_allocator& operator=(const cuda_default_allocator&) = default;

  char* allocate(std::size_t size) {
    // Set the device before allocation
    cudaError_t error = ::cudaSetDevice(device_id);
    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("cudaSetDevice({}) failed with error: {}",
                               device_id, cudaGetErrorString(error)));
    }

    void* ptr = nullptr;
    error = ::cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("cudaMalloc({} bytes) on device {} failed with error: {}",
                               size, device_id, cudaGetErrorString(error)));
    }

    return static_cast<char*>(ptr);
  }

  void deallocate(char* ptr, std::size_t /* size */) noexcept {
    // cudaFree doesn't require size parameter
    // Set device before deallocation
    cudaError_t error = ::cudaSetDevice(device_id);
    if (error != cudaSuccess) {
      // Cannot throw in deallocate, but we should handle this gracefully
      // In production, this would be logged
      return;
    }

    error = ::cudaFree(ptr);
    // Cannot throw in deallocate - this must be noexcept
    // In production, errors would be logged but not thrown
  }
};

// CUDA device memory resource implementation
//
// Template parameters:
// - Allocator: Underlying allocator (default: cuda_default_allocator)
// - Tracking: Enable allocation tracking (default: true)
//
// Usage:
// - Default CUDA allocations: Use cuda_device_memory::get() singleton (device 0)
// - Specific device: cuda_device_memory("GPU_1", 1) for device 1
// - Multi-GPU: Create separate instances for each device
// - High-frequency allocations: Consider wrapping with fixed_pool or quick_pool
// - Zero overhead needed: Use cuda_device_memory<cuda_default_allocator, false>
template<
  typename Allocator = cuda_default_allocator,
  bool Tracking = true
>
class cuda_device_memory : public memory_resource<cuda_platform, Allocator, Tracking> {
private:
  using base = memory_resource<cuda_platform, Allocator, Tracking>;

  int device_id_;

  // Singleton instance (for default config only - device 0)
  static cuda_device_memory& instance() {
    static cuda_device_memory inst;
    return inst;
  }

  // Private constructor for singleton (device 0)
  cuda_device_memory()
    : base("CUDA", cuda_default_allocator(0))
    , device_id_(0)
  {}

public:
  // Singleton access (device 0)
  static cuda_device_memory& get() {
    return instance();
  }

  // Allow custom instances for specific devices
  explicit cuda_device_memory(const std::string& name, int device_id = 0,
                               Allocator alloc = Allocator())
    : base(name, std::move(alloc))
    , device_id_(device_id)
  {
    // Verify device exists and is valid
    int device_count = 0;
    cudaError_t error = ::cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("cudaGetDeviceCount() failed with error: {}",
                               cudaGetErrorString(error)));
    }

    if (device_id < 0 || device_id >= device_count) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("Invalid CUDA device ID: {} (available devices: 0-{})",
                               device_id, device_count - 1));
    }
  }

  // Allow custom instances with just device_id
  explicit cuda_device_memory(int device_id)
    : cuda_device_memory(fmt::format("CUDA_{}", device_id), device_id,
                         cuda_default_allocator(device_id))
  {}

  // Get the device ID for this resource
  int get_device_id() const {
    return device_id_;
  }

  // Implement pure virtual from memory
  // Allocates CUDA device memory of the specified size
  //
  // @param size Number of bytes to allocate (0 returns nullptr)
  // @return Pointer to allocated device memory (never null for non-zero size)
  // @throws out_of_memory_error if allocation fails
  void* allocate(std::size_t size) override {
    if (size == 0) {
      return nullptr;  // Match cudaMalloc behavior
    }

    void* ptr = base::allocator_.allocate(size);

    if (!ptr) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("cuda_device_memory: allocation of {} bytes on device {} failed",
                               size, device_id_));
    }

    if constexpr (Tracking) {
      base::track_allocation(ptr, size);
    }

    return ptr;
  }

  // Deallocates memory previously allocated by this resource
  // MUST be noexcept as required by the interface
  //
  // @param ptr Pointer to deallocate (nullptr is safe no-op)
  void deallocate(void* ptr) override {
    if (!ptr) return;  // nullptr deallocation is safe no-op

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    // cuda_default_allocator::deallocate is noexcept
    base::allocator_.deallocate(static_cast<char*>(ptr), 0);
  }
};

// Convenience aliases
using default_cuda_device_memory = cuda_device_memory<cuda_default_allocator, true>;
using fast_cuda_device_memory = cuda_device_memory<cuda_default_allocator, false>;  // No tracking overhead

} // namespace resource
} // namespace umpire

// Provide cuda_allocator in umpire namespace for forward declaration compatibility
namespace umpire {
using cuda_allocator = resource::cuda_default_allocator;
}

#endif // UMPIRE_ENABLE_CUDA

#endif // UMPIRE_resource_cuda_device_memory_HPP
