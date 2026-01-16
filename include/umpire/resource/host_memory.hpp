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

// Helper: malloc-based allocator (doesn't require size in deallocate)
// This avoids the issue where std::allocator::deallocate() requires size
// but our deallocate() interface doesn't provide it.
struct malloc_allocator {
  char* allocate(std::size_t size) {
    return static_cast<char*>(std::malloc(size));
  }

  void deallocate(char* ptr, std::size_t /* size */) {
    std::free(ptr);
  }
};

// Host memory resource implementation
//
// Template parameters:
// - Allocator: Underlying allocator (default: malloc_allocator for simplicity)
// - Tracking: Enable allocation tracking (default: true)
//
// Usage:
// - Default host allocations: Use host_memory::get() singleton
// - High-frequency allocations: Consider wrapping with fixed_pool or quick_pool
// - Multi-threaded: Wrap with thread_safe<host_memory>
// - Zero overhead needed: Use host_memory<malloc_allocator, false> to disable tracking
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
  // Singleton access
  static host_memory& get() {
    return instance();
  }

  // Allow custom instances (for testing, custom allocators)
  explicit host_memory(const std::string& name, Allocator alloc = Allocator())
    : base(name, std::move(alloc)) {}

  // Implement pure virtual from memory
  // Allocates host memory of the specified size
  //
  // @param size Number of bytes to allocate (0 returns nullptr)
  // @return Pointer to allocated memory (never null for non-zero size)
  // @throws out_of_memory_error if allocation fails
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

  // Deallocates memory previously allocated by this resource
  //
  // @param ptr Pointer to deallocate (nullptr is safe no-op)
  void deallocate(void* ptr) override {
    if (!ptr) return;  // nullptr deallocation is safe no-op

    if constexpr (Tracking) {
      base::untrack_allocation(ptr);
    }

    // malloc_allocator doesn't need size, just passes ignored parameter
    base::allocator_.deallocate(static_cast<char*>(ptr), 0);
  }
};

// Convenience aliases
using default_host_memory = host_memory<malloc_allocator, true>;
using fast_host_memory = host_memory<malloc_allocator, false>;  // No tracking overhead

} // namespace resource
} // namespace umpire

#endif // UMPIRE_resource_host_memory_HPP
