//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_openmp_target_memory_HPP
#define UMPIRE_resource_openmp_target_memory_HPP

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)

#include <omp.h>

#include <string>
#include <utility>

#include "fmt/format.h"
#include "umpire/memory_resource.hpp"
#include "umpire/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {

/*!
 * \brief Backend allocator wrapper for OpenMP target memory.
 *
 * The allocator binds allocations to an OpenMP target device selected by
 * device ordinal.
 */
struct omp_target_allocator {
  //! OpenMP target device ordinal used for allocation and free.
  int device_id;

  /*!
   * \brief Construct an allocator targeting an OpenMP device.
   *
   * \param device Target device ordinal. Defaults to the current OpenMP target
   *        default device.
   */
  explicit omp_target_allocator(int device = omp_get_default_device())
    : device_id(device)
  {
  }

  omp_target_allocator(const omp_target_allocator&) = default;
  omp_target_allocator& operator=(const omp_target_allocator&) = default;

  /*!
   * \brief Allocate target memory with `omp_target_alloc`.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to target memory, or `nullptr` if the runtime cannot
   *         satisfy the request.
   */
  char* allocate(std::size_t size)
  {
    return static_cast<char*>(omp_target_alloc(size, device_id));
  }

  /*!
   * \brief Deallocate target memory with `omp_target_free`.
   *
   * \param ptr Pointer returned by allocate().
   * \param Unused size parameter required by the allocator wrapper interface.
   */
  void deallocate(char* ptr, std::size_t /* size */) noexcept
  {
    omp_target_free(ptr, device_id);
  }
};

namespace resource {

/*!
 * \brief API v2 resource for OpenMP target allocations.
 *
 * The default singleton targets the current OpenMP default device. Additional
 * instances can bind to specific devices for explicit multi-device workflows.
 *
 * \tparam Allocator Backend allocator wrapper.
 * \tparam Tracking Whether allocations are recorded in the shared v2 registry.
 */
template<
  typename Allocator = omp_target_allocator,
  bool Tracking = true
>
class openmp_target_memory : public memory_resource<omp_target_platform, Allocator, Tracking> {
private:
  using base = memory_resource<omp_target_platform, Allocator, Tracking>;

  int device_id_;

  static openmp_target_memory& instance()
  {
    static openmp_target_memory inst;
    return inst;
  }

  openmp_target_memory()
    : base("OMP_TARGET", Allocator(omp_get_default_device()))
    , device_id_(omp_get_default_device())
  {
  }

public:
  //! Return the default singleton bound to the OpenMP default target device.
  static openmp_target_memory& get()
  {
    return instance();
  }

  /*!
   * \brief Construct a named OpenMP target resource.
   *
   * \param name Human-readable resource name.
   * \param device_id OpenMP target device ordinal to use.
   */
  explicit openmp_target_memory(const std::string& name, int device_id = omp_get_default_device())
    : base(name, Allocator(device_id))
    , device_id_(device_id)
  {
  }

  /*!
   * \brief Construct a resource named from its device ordinal.
   *
   * \param device_id OpenMP target device ordinal to use.
   */
  explicit openmp_target_memory(int device_id)
    : openmp_target_memory(fmt::format("OMP_TARGET_{}", device_id), device_id)
  {
  }

  //! Return the OpenMP target device ordinal used by this resource.
  int get_device_id() const
  {
    return device_id_;
  }

  /*!
   * \brief Allocate OpenMP target memory.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to target memory, or `nullptr` for a zero-byte request.
   *
   * \throws out_of_memory_error if the allocation cannot be satisfied.
   */
  void* allocate(std::size_t size) override
  {
    if (size == 0) {
      return nullptr;
    }

    void* ptr = base::allocator_.allocate(size);

    if (!ptr) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("openmp_target_memory: allocation of {} bytes on device {} failed",
                               size, device_id_));
    }

    if constexpr (Tracking) {
      base::track_allocation(ptr, size);
    }

    return ptr;
  }

  /*!
   * \brief Deallocate OpenMP target memory previously returned by this resource.
   *
   * \param ptr Pointer to release. `nullptr` is a no-op.
   */
  void deallocate(void* ptr) override
  {
    if (!ptr) return;

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    base::allocator_.deallocate(static_cast<char*>(ptr), 0);
  }
};

//! Tracking-enabled OpenMP target resource alias.
using default_openmp_target_memory = openmp_target_memory<omp_target_allocator, true>;
//! OpenMP target resource alias with tracking disabled.
using fast_openmp_target_memory = openmp_target_memory<omp_target_allocator, false>;

} // namespace resource
} // namespace umpire

#endif // UMPIRE_ENABLE_OPENMP_TARGET

#endif // UMPIRE_resource_openmp_target_memory_HPP
