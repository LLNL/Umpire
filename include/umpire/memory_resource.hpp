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

/*!
 * \brief Maps a platform tag to its default low-level allocator wrapper.
 *
 * API v2 resources select their allocator implementation through this trait so
 * callers can usually rely on the default template argument instead of naming
 * the backend allocator explicitly.
 *
 * \tparam Platform Platform tag such as `host_platform` or `cuda_platform`.
 */
template<typename Platform>
struct default_allocator_for;

//! \brief Default allocator for host resources.
template<>
struct default_allocator_for<host_platform> {
  using type = std::allocator<char>;
};

#if defined(UMPIRE_ENABLE_CUDA)
//! \brief Forward declaration for the default CUDA allocator wrapper.
struct cuda_allocator;

//! \brief Default allocator for CUDA device resources.
template<>
struct default_allocator_for<cuda_platform> {
  using type = cuda_allocator;
};
#endif

#if defined(UMPIRE_ENABLE_HIP)
//! \brief Forward declaration for the default HIP allocator wrapper (defined in hip_device_memory.hpp).
namespace resource {
struct hip_default_allocator;
}
using hip_allocator = resource::hip_default_allocator;

//! \brief Default allocator for HIP device resources.
template<>
struct default_allocator_for<hip_platform> {
  using type = hip_allocator;
};
#endif

#if defined(UMPIRE_ENABLE_SYCL)
//! \brief Forward declaration for the default SYCL allocator wrapper.
struct sycl_allocator;

//! \brief Default allocator for SYCL device resources.
template<>
struct default_allocator_for<sycl_platform> {
  using type = sycl_allocator;
};
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
//! \brief Forward declaration for the default OpenMP target allocator wrapper.
struct omp_target_allocator;

//! \brief Default allocator for OpenMP target resources.
template<>
struct default_allocator_for<omp_target_platform> {
  using type = omp_target_allocator;
};
#endif

/*!
 * \brief Common CRTP-style base for typed API v2 memory resources.
 *
 * `memory_resource` connects a compile-time platform tag with a concrete
 * allocator implementation and optional registry tracking. Derived resource
 * types provide the public allocation semantics while reusing the allocator
 * storage and platform reporting implemented here.
 *
 * Thread safety guarantees:
 * - Read-only operations such as `get_name()`, `get_id()`, and
 *   `get_platform()` are safe after construction.
 * - Allocation and deallocation are only as thread-safe as the derived
 *   resource and wrapped allocator implementation.
 *
 * \tparam Platform Platform tag (`host_platform`, `cuda_platform`, etc.).
 * \tparam Allocator Backend allocator used to satisfy raw allocation requests.
 * \tparam Tracking Whether allocations should be registered in the shared v2
 *         registry for introspection and interoperability.
 */
template<
  typename Platform,
  typename Allocator = typename default_allocator_for<Platform>::type,
  bool Tracking = true
>
class memory_resource : public memory {
public:
  //! \brief Compile-time platform tag propagated through composed types.
  using platform = Platform;
  //! \brief Low-level allocator type used by this resource.
  using allocator_type = Allocator;
  //! \brief Indicates whether allocation tracking is enabled for this resource.
  static constexpr bool tracking_enabled = Tracking;

protected:
  //! \brief Stored backend allocator instance.
  Allocator allocator_;

  /*!
   * \brief Allocate bytes through the backend allocator and optionally track them.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer returned by the backend allocator.
   */
  void* allocate_impl(std::size_t size) {
    void* ptr = static_cast<void*>(allocator_.allocate(size));
    if constexpr (Tracking) {
      track_allocation(ptr, size);
    }
    return ptr;
  }

  /*!
   * \brief Deallocate bytes through the backend allocator and optionally untrack them.
   *
   * \param ptr Pointer to release.
   * \param size Original byte count when required by the backend allocator.
   */
  void deallocate_impl(void* ptr, std::size_t size) {
    if constexpr (Tracking) {
      untrack_allocation(ptr);
    }
    // Backend allocator wrappers use a char-based interface.
    allocator_.deallocate(static_cast<char*>(ptr), size);
  }

public:
  /*!
   * \brief Construct a named resource around an allocator instance.
   *
   * \param name Human-readable resource name exposed through `memory`.
   * \param alloc Allocator object to use for future requests.
   */
  explicit memory_resource(const std::string& name, Allocator alloc = Allocator())
    : memory(name)
    , allocator_(std::move(alloc))
  {}

  /*!
   * \brief Report the runtime platform corresponding to `Platform`.
   *
   * \return The CAMP platform enum associated with the compile-time tag.
   */
  resource::Platform get_platform() const override {
    return platform_for<Platform>::value;
  }
};

} // namespace umpire

#endif // UMPIRE_memory_resource_HPP
