//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_shared_memory_HPP
#define UMPIRE_resource_shared_memory_HPP

#include <cstddef>
#include <memory>
#include <string>

#include "umpire/memory.hpp"
#include "umpire/platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief API v2 POSIX host shared memory resource.
 *
 * `shared_memory` is a port of the v1 `HostSharedMemoryResource`: on
 * construction it creates (or attaches to, if another process/instance
 * already created it) a POSIX shared memory segment via `shm_open()` and
 * `mmap()`. The segment is carved up by a simple intrusive free/used block
 * allocator (protected by a `PTHREAD_PROCESS_SHARED` mutex living in the
 * segment itself) so that allocations can be looked up by name from any
 * process attached to the same segment via `find_pointer_from_name()`.
 *
 * Design notes / deliberate deviations from the class-template shape used by
 * other API v2 resources (e.g. `host_memory<Allocator, Tracking>`):
 *
 * - This is a plain (non-template) class rather than a
 *   `memory_resource<Platform, Allocator, Tracking>` specialization. The
 *   segment bookkeeping (shared header, mutex, offset-based free/used
 *   lists) and the extra named-allocation API surface
 *   (`allocate(name, bytes)`, `find_pointer_from_name()`) don't fit the
 *   `Allocator`-wraps-`allocate(size)`/`deallocate(ptr,size)` shape that
 *   `memory_resource` expects, so `shared_memory` inherits directly from
 *   `umpire::memory` and implements its own storage management in a `.cpp`
 *   (pimpl) rather than being header-only.
 * - `platform` is still exposed as a type alias (`host_platform`) purely for
 *   API consistency with the templated resources (e.g. code that is generic
 *   over `Memory::platform`); `get_platform()` is a normal (non-template)
 *   override since there's no `Platform` template parameter to dispatch on.
 * - Tracking is a runtime constructor flag (`tracking`) instead of a
 *   compile-time `Tracking` template parameter. Since `shared_memory` is
 *   already a concrete class with a `.cpp`-implemented pimpl, adding a bool
 *   template parameter would force either duplicating the pimpl per
 *   instantiation or templating the whole implementation for no benefit;
 *   a constructor flag keeps the single `.cpp` translation unit simple.
 *   Default is `true`, matching `host_memory`'s tracking-enabled default.
 *
 * Anonymous `allocate(bytes)` behavior differs from v1's
 * `HostSharedMemoryResource::allocate()`, which unconditionally throws
 * because plain `HostSharedMemoryResource` doesn't itself synthesize names
 * (that name synthesis lives in a separate v1 wrapper, `NamingShim`, that
 * composes on top of any `AllocationStrategy`). Since `umpire::memory`
 * requires a concrete `allocate(std::size_t)` override, `shared_memory`
 * bakes that same "synthesize a unique internal name" behavior directly in
 * for anonymous allocations, so they succeed rather than always throwing.
 *
 * Not ported from v1 (explicitly out of scope): the MPI3 shared memory
 * variant and `get_communicator_for_allocator()`-style MPI communicator
 * support. This resource is POSIX-only.
 */
class shared_memory : public umpire::memory {
public:
  //! Compile-time platform tag, for API consistency with templated resources.
  using platform = host_platform;

  /*!
   * \brief Create or attach to a named POSIX shared memory segment.
   *
   * \param name Segment name. A leading '/' is added if missing (required by
   *        `shm_open()`). Keep this short: macOS enforces a tight limit
   *        (`PSHMNAMLEN`, historically 31 bytes including the leading '/')
   *        on shared memory segment names.
   * \param size Total segment size in bytes, including internal bookkeeping
   *        overhead (segment header + per-allocation block headers). Only
   *        honored by whichever caller creates the segment; a caller that
   *        attaches to an already-existing segment uses the size
   *        established by the creator.
   * \param tracking Whether allocations newly created (not merely attached
   *        to) through this resource should be registered in the shared
   *        API v2 registry. Defaults to `true`.
   *
   * \throws runtime_error if the segment cannot be created, opened, sized,
   *         or mapped.
   */
  explicit shared_memory(const std::string& name, std::size_t size, bool tracking = true);

  //! Unmap and unlink the shared memory segment.
  ~shared_memory() override;

  shared_memory(const shared_memory&) = delete;
  shared_memory& operator=(const shared_memory&) = delete;
  shared_memory(shared_memory&&) = delete;
  shared_memory& operator=(shared_memory&&) = delete;

  /*!
   * \brief Allocate an anonymous shared memory block.
   *
   * Synthesizes a unique internal name (`<segment-name>_alloc_<counter>`,
   * mirroring v1's `NamingShim` pattern) and delegates to the same
   * named-allocation path used by `allocate(name, bytes)`, since the
   * underlying segment only knows how to store named blocks.
   *
   * \param bytes Number of bytes to allocate. Returns `nullptr` for `0`.
   * \return Pointer to the allocated storage.
   *
   * \throws out_of_memory_error if no free block large enough is available.
   */
  void* allocate(std::size_t bytes) override;

  /*!
   * \brief Allocate a named shared memory block, or attach to it if a block
   *        with this name already exists.
   *
   * If `name` already identifies a live allocation in the segment (created
   * by this process or another process/instance attached to the same
   * segment), this call increments that allocation's reference count and
   * returns its existing pointer rather than creating a new block;
   * `deallocate()` must be called once per successful `allocate()` call
   * (matching calls decrement the reference count, and the block is only
   * released back to the free list once it reaches zero).
   *
   * \param name Allocation name, unique within this segment.
   * \param bytes Number of bytes to allocate for a *new* block. Ignored
   *        when attaching to an existing block of the same name.
   * \return Pointer to the allocation's memory.
   *
   * \throws out_of_memory_error if no free block large enough is available
   *         for a new allocation.
   */
  void* allocate(const std::string& name, std::size_t bytes);

  /*!
   * \brief Release a shared memory allocation.
   *
   * Decrements the allocation's reference count; the block is only
   * unlinked and returned to the segment's free list (and, if tracking is
   * enabled, deregistered from the v2 registry) once the reference count
   * reaches zero. `nullptr` is a safe no-op.
   *
   * \param ptr Pointer previously returned by `allocate()`.
   */
  void deallocate(void* ptr) override;

  /*!
   * \brief Look up a live allocation in this segment by name.
   *
   * \param name Allocation name to search for.
   * \return Pointer to the allocation's memory, or `nullptr` if no live
   *         allocation with this name exists in the segment.
   */
  void* find_pointer_from_name(const std::string& name);

  //! \brief Runtime platform tag; always `Platform::host`.
  resource::Platform get_platform() const override;

  /*!
   * \brief Total segment bytes currently accounted for (allocations plus
   *        per-allocation bookkeeping overhead), as tracked in the shared
   *        segment header. This reflects the state of the whole segment
   *        (visible to every process/instance attached to it), unlike
   *        `get_actual_size()` (inherited from `umpire::memory`), which
   *        only reflects bytes newly created through this instance.
   */
  std::size_t get_segment_actual_size() const;

private:
  class impl;
  std::unique_ptr<impl> pimpl_;
  bool tracking_;
};

} // namespace resource
} // namespace umpire

#endif // UMPIRE_resource_shared_memory_HPP
