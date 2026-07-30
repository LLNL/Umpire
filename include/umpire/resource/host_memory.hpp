//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_host_memory_HPP
#define UMPIRE_resource_host_memory_HPP

#include <cstdlib>
#include <string>

#include "umpire/memory_resource.hpp"
#include "umpire/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Default host allocator wrapper used by `host_memory`.
 *
 * This adapter exposes a `std::allocator`-like interface on top of
 * `std::malloc` and `std::free`, which keeps deallocation compatible with the
 * pointer-only API v2 `memory` interface.
 */
struct malloc_allocator {
  //! \brief Allocate `size` bytes of host memory with `std::malloc`.
  static char* allocate(std::size_t size) {
    return static_cast<char*>(std::malloc(size));
  }

  //! \brief Free memory returned from `allocate()`.
  static void deallocate(char* ptr, std::size_t /* size */) {
    std::free(ptr);
  }
};

/*!
 * \brief Concrete API v2 memory resource for ordinary host allocations.
 *
 * `host_memory` is the standard entry point for CPU-accessible API v2
 * allocations. It can be used directly, wrapped in strategies, or paired with
 * `umpire::allocator<T, Memory>` for standard-library containers.
 *
 * \tparam Allocator Backend allocator wrapper used for host allocation.
 * \tparam Tracking Whether allocations should be tracked in the shared v2
 *         registry.
 */
template<
  typename Allocator = malloc_allocator,
  bool Tracking = true
>
class host_memory : public memory_resource<host_platform, Allocator, Tracking> {
private:
  using base = memory_resource<host_platform, Allocator, Tracking>;

  // Singleton instance (for default config only)
  static host_memory& instance() {
    static host_memory inst;
    return inst;
  }

  // Private constructor for singleton
  host_memory() : base("HOST") {}

public:
  //! \brief Return the process-wide default HOST resource singleton.
  static host_memory& get() {
    return instance();
  }

  /*!
   * \brief Construct a named host resource with a custom allocator instance.
   *
   * \param name Human-readable resource name.
   * \param alloc Allocator object used to satisfy allocation requests.
   */
  explicit host_memory(const std::string& name, Allocator alloc = Allocator())
    : base(name, std::move(alloc)) {}

  /*!
   * \brief Allocate host memory.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to host-accessible storage, or `nullptr` for a zero-byte
   *         request.
   *
   * \throws out_of_memory_error if the allocation cannot be satisfied.
   */
  void* allocate(std::size_t size) override {
    if (size == 0) {
      return nullptr;  // Match std::malloc behavior
    }

    void* ptr = base::allocator_.allocate(size);

    if (!ptr) {
      UMPIRE_ERROR(out_of_memory_error,
                   fmt::format("host_memory: allocation of {} bytes failed", size));
    }

    if constexpr (Tracking) {
      base::track_allocation(ptr, size);
    }

    return ptr;
  }

  /*!
   * \brief Deallocate memory previously returned by this resource.
   *
   * \param ptr Pointer to release. `nullptr` is a no-op.
   */
  void deallocate(void* ptr) override {
    if (!ptr) return;  // nullptr deallocation is safe no-op

    // malloc_allocator ignores the size parameter
    base::deallocate_impl(ptr, 0);
  }
};

//! \brief Tracking-enabled host resource using `malloc_allocator`.
using default_host_memory = host_memory<malloc_allocator, true>;
//! \brief Host resource alias with tracking disabled for low-overhead paths.
using fast_host_memory = host_memory<malloc_allocator, false>;  // No tracking overhead

} // namespace resource
} // namespace umpire

#endif // UMPIRE_resource_host_memory_HPP
