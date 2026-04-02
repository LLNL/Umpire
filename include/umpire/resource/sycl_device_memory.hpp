//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_sycl_device_memory_HPP
#define UMPIRE_resource_sycl_device_memory_HPP

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_SYCL)

#include <string>
#include <utility>

#include "fmt/format.h"
#include "umpire/memory_resource.hpp"
#include "umpire/platform.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/sycl_compat.hpp"

namespace umpire {

/*!
 * \brief Backend allocator wrapper for SYCL device memory.
 *
 * The allocator owns a SYCL queue and uses it for all allocation and
 * deallocation requests.
 */
struct sycl_allocator {
  //! Queue used for SYCL allocation and free operations.
  sycl::queue queue;

  //! Construct an allocator with a default-constructed queue.
  sycl_allocator() = default;

  /*!
   * \brief Construct an allocator from an existing queue.
   *
   * \param q Queue used for subsequent allocation and deallocation.
   */
  explicit sycl_allocator(sycl::queue q)
    : queue(std::move(q))
  {
  }

  sycl_allocator(const sycl_allocator&) = default;
  sycl_allocator& operator=(const sycl_allocator&) = default;

  /*!
   * \brief Allocate device memory with `sycl::malloc_device`.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to device memory.
   *
   * \throws runtime_error if the SYCL backend reports an error.
   */
  char* allocate(std::size_t size)
  {
    try {
      return static_cast<char*>(sycl::malloc_device(size, queue));
    } catch (const sycl::exception& e) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("sycl::malloc_device({} bytes) failed: {}", size, e.what()));
    }
  }

  /*!
   * \brief Deallocate device memory with `sycl::free`.
   *
   * \param ptr Pointer returned by allocate().
   * \param size Unused size parameter required by the allocator wrapper
   *        interface.
   */
  void deallocate(char* ptr, std::size_t /* size */)
  {
    sycl::free(ptr, queue);
  }
};

namespace resource {

//! Alias for the default SYCL allocator wrapper.
using sycl_default_allocator = sycl_allocator;

/*!
 * \brief API v2 resource for SYCL device allocations.
 *
 * Instances bind to a SYCL queue supplied by the caller. That queue is also
 * exposed through accessors so higher-level code can launch kernels and copies
 * on the same execution context.
 *
 * \tparam Allocator Backend allocator wrapper.
 * \tparam Tracking Whether allocations are recorded in the shared v2 registry.
 */
template<
  typename Allocator = sycl_allocator,
  bool Tracking = true
>
class sycl_device_memory : public memory_resource<sycl_platform, Allocator, Tracking> {
private:
  using base = memory_resource<sycl_platform, Allocator, Tracking>;

  sycl::queue queue_;

public:
  /*!
   * \brief Construct a named SYCL resource bound to a queue.
   *
   * \param name Human-readable resource name.
   * \param queue Queue used for memory operations.
   */
  explicit sycl_device_memory(const std::string& name, sycl::queue queue)
    : base(name, Allocator(queue))
    , queue_(std::move(queue))
  {
  }

  //! Access the mutable queue associated with this resource.
  sycl::queue& get_queue()
  {
    return queue_;
  }

  //! Access the queue associated with this resource.
  const sycl::queue& get_queue() const
  {
    return queue_;
  }

  /*!
   * \brief Allocate SYCL device memory.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to device memory, or `nullptr` for a zero-byte request.
   *
   * \throws runtime_error if the SYCL backend reports an error.
   * \throws out_of_memory_error if allocation returns a null pointer.
   */
  void* allocate(std::size_t size) override
  {
    if (size == 0) {
      return nullptr;
    }

    void* ptr = nullptr;

    try {
      ptr = base::allocator_.allocate(size);
    } catch (const sycl::exception& e) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("sycl_device_memory: allocation of {} bytes failed: {}",
                               size, e.what()));
    }

    if (!ptr) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("sycl_device_memory: allocation of {} bytes failed", size));
    }

    if constexpr (Tracking) {
      base::track_allocation(ptr, size);
    }

    return ptr;
  }

  /*!
   * \brief Deallocate SYCL device memory previously returned by this resource.
   *
   * \param ptr Pointer to release. `nullptr` is a no-op.
   *
   * \throws runtime_error if `sycl::free` reports an error.
   */
  void deallocate(void* ptr) override
  {
    if (!ptr) return;

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    try {
      base::allocator_.deallocate(static_cast<char*>(ptr), 0);
    } catch (const sycl::exception& e) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("sycl::free(ptr={}) failed: {}",
                               ptr, e.what()));
    }
  }
};

//! Tracking-enabled SYCL device resource alias.
using default_sycl_device_memory = sycl_device_memory<sycl_default_allocator, true>;
//! SYCL device resource alias with tracking disabled.
using fast_sycl_device_memory = sycl_device_memory<sycl_default_allocator, false>;

} // namespace resource
} // namespace umpire

#endif // UMPIRE_ENABLE_SYCL

#endif // UMPIRE_resource_sycl_device_memory_HPP
