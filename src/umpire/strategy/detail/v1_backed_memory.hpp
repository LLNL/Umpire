//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_detail_v1_backed_memory_HPP
#define UMPIRE_strategy_detail_v1_backed_memory_HPP

#include <cstddef>
#include <string>

#include "umpire/allocation_record.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/memory.hpp"
#include "umpire/resource/platform.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {
namespace detail {

/*!
 * \brief Bridges a v1 `strategy::AllocationStrategy*` parent so it can be
 * used as the `Memory` parent of an API v2 `strategy::allocation_strategy`
 * template (e.g. `size_limiter<Memory>`, `named<Memory>`,
 * `thread_safe<Memory>`).
 *
 * This is the mirror image of the existing HOST-only v1-on-v2 bridge in
 * `umpire::memory` (see `should_bridge_to_v1_host_allocator` in
 * src/umpire/memory.cpp): here a v1 `AllocationStrategy` is the parent and a
 * v2 `memory` object is the (synthetic, non-owning) child that forwards
 * allocate()/deallocate() calls onto it.
 *
 * \par Tracking / statistics
 * `v1_backed_memory` intentionally does NOT call `memory::track_allocation()`
 * / `untrack_allocation()`. Doing so would emit a second, redundant
 * `event::allocate` / `event::deallocate` pair (the outer v1
 * `Allocator::do_allocate`/`do_deallocate`, see `src/umpire/Allocator.inl`,
 * already emits one referencing the v1 strategy) and would double-count
 * live-byte statistics that the v1 `AllocationStrategy::allocate_internal()`
 * / `deallocate_internal()` path already maintains on `m_current_size` /
 * `m_high_watermark` / `m_allocation_count` (invoked here, and by every other
 * v1 strategy that forwards to a parent, e.g. QuickPool's
 * `m_allocator->allocate_internal(size)` / `deallocate_internal(buffer,
 * size)`).
 *
 * \par Registry visibility
 * Despite not calling `track_allocation()`, this bridge still registers each
 * live allocation directly with the shared API v2 registry
 * (`detail::registry::register_allocation()` / `remove_allocation()`, which
 * do not emit events or touch `memory`'s atomic statistics) so that
 * allocations made through a delegated v1 strategy remain discoverable via
 * `detail::registry::find_allocations_by_memory()` / `find_allocation()`,
 * matching the interoperability guarantees the HOST v1<->v2 bridge already
 * provides in the other direction. The registered size doubles as the
 * source of truth this class needs to recover a size for
 * `deallocate(void*)`, since the v2 `memory` interface carries no size on
 * that call.
 *
 * \par Platform
 * v1 and v2 share the same runtime platform enum
 * (`camp::resources::Platform`, aliased as both `umpire::Platform` and
 * `umpire::resource::Platform`), so `get_platform()` simply forwards to the
 * v1 parent's `getPlatform()`. There is, however, no compile-time `platform`
 * type this class could expose (the v1 parent's platform is only known at
 * runtime), so `v1_backed_memory` deliberately omits a `platform` member
 * type alias. `strategy::thread_safe<Memory>` already tolerates a missing
 * `Memory::platform` via its `detail::thread_safe_platform` SFINAE helper;
 * `size_limiter<Memory>` and `named<Memory>` are relaxed with the same
 * pattern under `UMPIRE_V1_DELEGATE_TO_V2` (see
 * include/umpire/strategy/size_limiter.hpp and
 * include/umpire/strategy/named.hpp) so they can wrap this class too.
 *
 * Header placement note: this lives under src/ rather than include/ because
 * it is purely an implementation detail used to wire the v1
 * strategy::AllocationStrategy .cpp files (which already live under src/)
 * to the v2 strategy templates; it is not part of the public API v2
 * surface and should not be installed.
 */
class v1_backed_memory : public umpire::memory {
public:
  explicit v1_backed_memory(AllocationStrategy* v1_parent)
    : umpire::memory{v1_parent ? v1_parent->getName() : std::string{"<null>"}}
    , v1_parent_{v1_parent}
  {
  }

  void* allocate(std::size_t bytes) override
  {
    void* ptr = v1_parent_->allocate_internal(bytes);

    if (ptr) {
      umpire::detail::registry::get().register_allocation(umpire::allocation_record{ptr, bytes, this});
    }

    return ptr;
  }

  void deallocate(void* ptr) override
  {
    if (!ptr) {
      return;
    }

    std::size_t size = 0;
    if (auto record = umpire::detail::registry::get().find_allocation(ptr)) {
      size = record->size;
    }

    umpire::detail::registry::get().remove_allocation(ptr);
    v1_parent_->deallocate_internal(ptr, size);
  }

  resource::Platform get_platform() const override
  {
    return v1_parent_->getPlatform();
  }

  //! \brief Opt out of the v2 "fast path" reallocate()/move()/deallocate()
  //! dispatch (see umpire::memory::supports_v2_fast_path()).
  //!
  //! Allocations made through this bridge are mirrored into the shared v2
  //! registry (see allocate()/deallocate() above) purely for read-only
  //! discoverability (find_allocation()/find_allocations_by_memory()); the
  //! authoritative bookkeeping for these allocations is still v1's
  //! ResourceManager::m_allocations, updated by the OUTER v1
  //! Allocator::do_allocate()/do_deallocate() (see src/umpire/Allocator.inl)
  //! that wraps this bridge. If fast-path callers (ResourceManager::
  //! reallocate()/move()/deallocate(), op::dispatch's reallocate<T>()) were
  //! to call allocate()/deallocate() on this object directly, they would
  //! update the v2 registry but skip v1's registerAllocation()/
  //! deregisterAllocation(), silently desynchronizing m_allocations from
  //! the real set of live pointers.
  bool supports_v2_fast_path() const override
  {
    return false;
  }

  //! \brief Access the wrapped v1 parent strategy.
  AllocationStrategy* v1_parent() const
  {
    return v1_parent_;
  }

private:
  AllocationStrategy* v1_parent_;
};

} // namespace detail
} // namespace strategy
} // namespace umpire

#endif // UMPIRE_strategy_detail_v1_backed_memory_HPP
