//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_memory_resource_HPP
#define UMPIRE_memory_resource_HPP

#include <memory>
#include <string>
#include <utility>

#include "umpire/memory.hpp"
#include "umpire/platform.hpp"

namespace umpire {

// Forward declaration of default allocator metafunction
template<typename Platform>
struct default_allocator_for;

// Default allocator selection based on platform
template<>
struct default_allocator_for<host_platform> {
  using type = std::allocator<char>;
};

#if defined(UMPIRE_ENABLE_CUDA)
// Forward declaration - will be defined in cuda_device_memory.hpp
struct cuda_allocator;

template<>
struct default_allocator_for<cuda_platform> {
  using type = cuda_allocator;
};
#endif

#if defined(UMPIRE_ENABLE_HIP)
// Forward declaration - will be defined in hip_device_memory.hpp
struct hip_allocator;

template<>
struct default_allocator_for<hip_platform> {
  using type = hip_allocator;
};
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
// Forward declaration - will be defined in openmp_target_memory.hpp
struct omp_target_allocator;

template<>
struct default_allocator_for<omp_target_platform> {
  using type = omp_target_allocator;
};
#endif

// Template parameters:
// - Platform: Type tag (host_platform, cuda_platform, etc.)
// - Allocator: Underlying allocation mechanism (default varies by platform)
// - Tracking: Enable/disable allocation tracking (default true)
template<
  typename Platform,
  typename Allocator = typename default_allocator_for<Platform>::type,
  bool Tracking = true
>
class memory_resource : public memory {
public:
  // Type aliases for propagation through templates
  using platform = Platform;
  using allocator_type = Allocator;
  static constexpr bool tracking_enabled = Tracking;

protected:
  Allocator allocator_;  // Underlying allocator (e.g., std::allocator, cudaMalloc wrapper)

  // Helper for derived classes to conditionally track
  void* allocate_impl(std::size_t size) {
    using value_type = typename std::allocator_traits<Allocator>::value_type;
    void* ptr = static_cast<void*>(allocator_.allocate(size));
    if constexpr (Tracking) {
      track_allocation(ptr, size);
    }
    return ptr;
  }

  void deallocate_impl(void* ptr, std::size_t size) {
    if constexpr (Tracking) {
      untrack_allocation(ptr);
    }
    using value_type = typename std::allocator_traits<Allocator>::value_type;
    allocator_.deallocate(static_cast<value_type*>(ptr), size);
  }

public:
  // Constructor
  explicit memory_resource(const std::string& name, Allocator alloc = Allocator())
    : memory(name)
    , allocator_(std::move(alloc))
  {}

  // Implement pure virtual from memory base
  resource::Platform get_platform() const override {
    return platform_for<Platform>::value;
  }

  // Note: allocate() and deallocate() still pure virtual
  // Concrete resources (host_memory, cuda_device_memory) will implement
};

} // namespace umpire

#endif // UMPIRE_memory_resource_HPP
