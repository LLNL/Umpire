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

struct sycl_allocator {
  sycl::queue queue;

  sycl_allocator() = default;
  explicit sycl_allocator(sycl::queue q)
    : queue(std::move(q))
  {
  }

  sycl_allocator(const sycl_allocator&) = default;
  sycl_allocator& operator=(const sycl_allocator&) = default;

  char* allocate(std::size_t size)
  {
    try {
      return static_cast<char*>(sycl::malloc_device(size, queue));
    } catch (const sycl::exception& e) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("sycl::malloc_device({} bytes) failed: {}", size, e.what()));
    }
  }

  void deallocate(char* ptr, std::size_t /* size */)
  {
    sycl::free(ptr, queue);
  }
};

namespace resource {

using sycl_default_allocator = sycl_allocator;

template<
  typename Allocator = sycl_allocator,
  bool Tracking = true
>
class sycl_device_memory : public memory_resource<sycl_platform, Allocator, Tracking> {
private:
  using base = memory_resource<sycl_platform, Allocator, Tracking>;

  sycl::queue queue_;

public:
  explicit sycl_device_memory(const std::string& name, sycl::queue queue)
    : base(name, Allocator(queue))
    , queue_(std::move(queue))
  {
  }

  sycl::queue& get_queue()
  {
    return queue_;
  }

  const sycl::queue& get_queue() const
  {
    return queue_;
  }

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

using default_sycl_device_memory = sycl_device_memory<sycl_default_allocator, true>;
using fast_sycl_device_memory = sycl_device_memory<sycl_default_allocator, false>;

} // namespace resource
} // namespace umpire

#endif // UMPIRE_ENABLE_SYCL

#endif // UMPIRE_resource_sycl_device_memory_HPP
