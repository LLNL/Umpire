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

// OpenMP target allocator wrapper that uses omp_target_alloc/omp_target_free.
// Doesn't require size in deallocate.
struct omp_target_allocator {
  int device_id;

  explicit omp_target_allocator(int device = omp_get_default_device())
    : device_id(device)
  {
  }

  omp_target_allocator(const omp_target_allocator&) = default;
  omp_target_allocator& operator=(const omp_target_allocator&) = default;

  char* allocate(std::size_t size)
  {
    return static_cast<char*>(omp_target_alloc(size, device_id));
  }

  void deallocate(char* ptr, std::size_t /* size */) noexcept
  {
    omp_target_free(ptr, device_id);
  }
};

namespace resource {

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
  static openmp_target_memory& get()
  {
    return instance();
  }

  explicit openmp_target_memory(const std::string& name, int device_id = omp_get_default_device())
    : base(name, Allocator(device_id))
    , device_id_(device_id)
  {
  }

  explicit openmp_target_memory(int device_id)
    : openmp_target_memory(fmt::format("OMP_TARGET_{}", device_id), device_id)
  {
  }

  int get_device_id() const
  {
    return device_id_;
  }

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

  void deallocate(void* ptr) override
  {
    if (!ptr) return;

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    base::allocator_.deallocate(static_cast<char*>(ptr), 0);
  }
};

using default_openmp_target_memory = openmp_target_memory<omp_target_allocator, true>;
using fast_openmp_target_memory = openmp_target_memory<omp_target_allocator, false>;

} // namespace resource
} // namespace umpire

#endif // UMPIRE_ENABLE_OPENMP_TARGET

#endif // UMPIRE_resource_openmp_target_memory_HPP
