//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_v2_backed_resource_HPP
#define UMPIRE_resource_v2_backed_resource_HPP

#include <functional>
#include <memory>
#include <string>

#include "umpire/memory.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Bridges an API v2 `umpire::memory` object so it can be used as a v1
 * `resource::MemoryResource` (i.e. as the concrete implementation returned
 * from a v1 `MemoryResourceFactory::create()`).
 *
 * This is the mirror image of `strategy::detail::v1_backed_memory` (see
 * src/umpire/strategy/detail/v1_backed_memory.hpp): here a v2 `umpire::memory`
 * object is the parent doing the real allocation work, and this class is the
 * (owning) v1 `MemoryResource` child that forwards `allocate()`/
 * `deallocate()` calls onto it.
 *
 * \par Tracking / double-tracking hazard
 * v1's `Allocator::do_allocate()`/`do_deregisterAllocation()` (see
 * src/umpire/Allocator.inl) already registers every user allocation made
 * through a v1 `Allocator` in `ResourceManager::m_allocations` -- this is the
 * SAME bookkeeping path used by every other v1 resource/strategy. If the
 * wrapped v2 `memory` object also tracked the same allocation (i.e. called
 * `memory::track_allocation()`), the allocation would be registered TWICE:
 * once by v1 (authoritative) and once in the v2 registry (`detail::registry`)
 * -- and, for an object literally named "HOST", a THIRD time via the
 * HOST-only v1 bridge in `src/umpire/memory.cpp`
 * (`should_bridge_to_v1_host_allocator()` triggers whenever
 * `get_name() == "HOST" && get_platform() == Platform::host`, and would call
 * back into `rm.registerAllocation()`, i.e. try to double-register the same
 * pointer with v1).
 *
 * To avoid this, factories that delegate to this adapter MUST wrap a v2
 * resource instance constructed with:
 *   1. `Tracking = false` (e.g. `resource::fast_host_memory`,
 *      `resource::cuda_device_memory<Allocator, false>`, ...), so the v2
 *      object never calls `track_allocation()`/`untrack_allocation()` at all,
 *      and
 *   2. a name that is NOT exactly "HOST" (to also avoid the HOST bridge on
 *      the off chance a future Tracking=true instance is ever substituted
 *      here) -- factories pass a name derived from the v1-supplied name
 *      (e.g. suffixed with "_v2backed") to the wrapped v2 object's
 *      constructor while the v1-visible name (returned by
 *      `AllocationStrategy::getName()`) remains exactly what the caller
 *      requested (typically "HOST").
 *
 * With Tracking=false, v1's `ResourceManager::m_allocations` remains the
 * SOLE authoritative bookkeeping system for allocations made through a
 * delegated resource -- exactly as in the non-delegated (native) build.
 * (Contrast this with `v1_backed_memory`, which -- in the opposite direction
 * -- mirrors allocations into the v2 registry for read-only discoverability
 * because there the v1 side is authoritative and the v2 registry is the
 * secondary system. Here, the v2 side would be the *non*-authoritative one,
 * and since v2's `memory` base class does not expose an "insert into the
 * registry without touching statistics" primitive independent of
 * `track_allocation()`, the simplest and safest option is to disable v2
 * tracking entirely for delegated resources rather than partially
 * duplicating v1_backed_memory's bespoke bookkeeping in reverse.)
 *
 * \par Statistics
 * Exactly as with `v1_backed_memory`, the v1
 * `AllocationStrategy::allocate_internal()`/`deallocate_internal()` call
 * sequence (which updates `m_current_size`/`m_high_watermark`/
 * `m_allocation_count`) is untouched by this class: `MemoryResource`
 * (this class's base) is itself a `strategy::AllocationStrategy`, and
 * `allocate_internal()`/`deallocate_internal()` wrap the *private*
 * `allocate()`/`deallocate()` virtuals implemented here. So
 * `getCurrentSize()`/`getHighWatermark()`/`getAllocationCount()` behave
 * identically to the native (non-delegated) path.
 *
 * \par Accessibility
 * v2 `umpire::memory` has no equivalent of v1's
 * `MemoryResource::isAccessibleFrom(Platform)` (used by
 * `Umpire.cpp::is_accessible()`), so this adapter cannot simply forward that
 * query. Instead, each factory supplies an `is_accessible_from` callback that
 * replicates the accessibility rule of the v1 allocator wrapper it is
 * replacing (e.g. `alloc::MallocAllocator::isAccessible()` for HOST), so
 * behavior stays identical to the native path.
 *
 * \par Ownership
 * This adapter owns the wrapped v2 `memory` object via `unique_ptr`. v1's
 * `ResourceManager` owns/destroys `MemoryResource` instances (and therefore
 * this adapter, and therefore the wrapped v2 object) at shutdown, exactly as
 * it does for every other native `MemoryResource`. Factories must construct
 * a NAMED v2 instance (e.g. `resource::fast_host_memory{"HOST_v2backed"}`)
 * rather than reusing a v2 singleton (`host_memory::get()`), to avoid a
 * lifetime/name collision with any other consumer of that singleton.
 */
class v2_backed_resource : public MemoryResource {
 public:
  //! \brief Callback replicating a v1 allocator wrapper's `isAccessible()`.
  using AccessibilityFn = std::function<bool(Platform)>;

  /*!
   * \param name Name exposed through the v1 `AllocationStrategy` interface
   *        (i.e. what `rm.getAllocator(name)` will look up).
   * \param id Unique v1 allocator id (see `ResourceManager::getNextId()`).
   * \param traits Traits computed by the v1 factory; v2 has no equivalent
   *        concept, so this adapter stores and returns them itself (via the
   *        `MemoryResource` base).
   * \param platform Runtime platform reported by `getPlatform()`.
   * \param v2_memory The (Tracking=false, distinctly-named) v2 `memory`
   *        object that performs the real allocation work. Must be non-null.
   * \param is_accessible_from Replicates the wrapped v1 allocator's
   *        `isAccessible(Platform)` semantics.
   */
  v2_backed_resource(const std::string& name, int id, MemoryResourceTraits traits, Platform platform,
                      std::unique_ptr<umpire::memory> v2_memory, AccessibilityFn is_accessible_from);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;

  Platform getPlatform() noexcept override;
  bool isAccessibleFrom(Platform p) noexcept override;

  //! \brief Access the wrapped v2 memory object.
  umpire::memory* v2_memory() const noexcept
  {
    return m_v2_memory.get();
  }

 private:
  Platform m_platform;
  std::unique_ptr<umpire::memory> m_v2_memory;
  AccessibilityFn m_is_accessible_from;
};

} // namespace resource
} // namespace umpire

#endif // UMPIRE_resource_v2_backed_resource_HPP
