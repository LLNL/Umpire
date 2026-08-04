//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/memory.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/allocation_record.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/error.hpp"
#include "umpire/event/event.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Macros.hpp"

#include "fmt/format.h"

namespace umpire {

namespace {

//! \brief The canonical v2 HOST singleton (`resource::host_memory<>::get()`)
//! must always mirror into v1's ResourceManager, independent of
//! `memory::mirrors_to_v1()`, to preserve existing v1/v2 interop behavior
//! (see tests/integration/api_v2/test_v1_v2_interop.cpp).
bool should_bridge_to_v1_host_allocator(const memory& mem)
{
  return mem.get_name() == "HOST" && mem.get_platform() == resource::Platform::host;
}

//! \brief Resolve the v1 `AllocationStrategy*` that should be recorded on a
//! mirrored `util::AllocationRecord` for `mem`.
//!
//! Prefers a v1 allocator registered under `mem`'s own name (so a mirrored
//! object that shares a name with a v1 resource, e.g. "HOST", reuses that
//! resource's v1 strategy). Falls back to the v1 "HOST" allocator's
//! strategy for host-platform objects that have no matching v1 allocator by
//! name. Returns `nullptr` if neither resolves, meaning the caller should
//! not mirror the allocation, since there would be no sensible v1 strategy
//! pointer to record.
strategy::AllocationStrategy* resolve_v1_mirror_strategy(const memory& mem)
{
  auto& rm = ResourceManager::getInstance();

  if (rm.isAllocator(mem.get_name())) {
    return rm.getAllocator(mem.get_name()).getAllocationStrategy();
  }

  if (mem.get_platform() == resource::Platform::host && rm.isAllocator("HOST")) {
    return rm.getAllocator("HOST").getAllocationStrategy();
  }

  return nullptr;
}

void register_with_v1_allocator(const memory& mem, void* ptr, std::size_t size)
{
  auto& rm = ResourceManager::getInstance();
  if (rm.hasAllocator(ptr)) {
    return;
  }

  auto* strategy = resolve_v1_mirror_strategy(mem);
  if (!strategy) {
    UMPIRE_LOG(Debug, "memory \"" << mem.get_name()
                                  << "\" opted into v1 mirroring but no matching v1 allocator "
                                     "could be resolved; skipping mirror for ptr="
                                  << ptr);
    return;
  }

  rm.registerAllocation(ptr, util::AllocationRecord{ptr, size, strategy});
}

void deregister_from_v1_allocator(void* ptr)
{
  auto& rm = ResourceManager::getInstance();
  if (rm.hasAllocator(ptr)) {
    rm.deregisterAllocation(ptr);
  }
}

} // namespace

memory::memory(const std::string& name)
  : id_{detail::registry::get().get_id()}
  , name_{name}
{
  detail::registry::get().register_allocator(this);
}

memory::~memory()
{
  auto live_allocations = detail::registry::get().find_allocations_by_memory(this);
  if (!live_allocations.empty()) {
    std::size_t live_bytes{0};
    for (const auto& record : live_allocations) {
      live_bytes += record.size;
    }
    UMPIRE_LOG(Warning, "memory \"" << name_ << "\" (id=" << id_ << ") destroyed with "
                                    << live_allocations.size() << " active allocation(s) totaling "
                                    << live_bytes << " bytes; these will not be deallocated");
  }

  detail::registry::get().deregister_allocator(this);
}

void memory::track_allocation(void* ptr, std::size_t size)
{
  allocation_record record{ptr, size, this};
  detail::registry::get().register_allocation(record);
  if (mirrors_to_v1() || should_bridge_to_v1_host_allocator(*this)) {
    register_with_v1_allocator(*this, ptr, size);
  }
  update_statistics(static_cast<std::ptrdiff_t>(size));

  umpire::event::record<umpire::event::allocate>(
      [&](auto& event) { event.size(size).ref(static_cast<void*>(this)).ptr(ptr); });
}

void memory::untrack_allocation(void* ptr)
{
  auto record = detail::registry::get().find_allocation(ptr);
  if (!record) {
    throw unknown_allocation(fmt::format("Attempted to deallocate unknown pointer {:p}", ptr));
  }

  if (record->strategy != this) {
    throw unknown_allocation(fmt::format(
        "Attempted to deallocate pointer {:p} through memory \"{}\" (id={}), but it is owned by \"{}\" (id={})", ptr,
        name_, id_, record->strategy ? record->strategy->get_name() : "<unknown>",
        record->strategy ? record->strategy->get_id() : -1));
  }

  std::size_t size = record->size;

  umpire::event::record<umpire::event::deallocate>(
      [&](auto& event) { event.ref(static_cast<void*>(this)).ptr(ptr); });

  detail::registry::get().remove_allocation(ptr);
  if (mirrors_to_v1() || should_bridge_to_v1_host_allocator(*this)) {
    deregister_from_v1_allocator(ptr);
  }
  update_statistics(-static_cast<std::ptrdiff_t>(size));
}

void memory::update_statistics(std::ptrdiff_t size_delta)
{
  update_current_size(size_delta);
  update_actual_size(size_delta);
}

void memory::update_current_size(std::ptrdiff_t size_delta)
{
  std::size_t old_current = current_size_.fetch_add(size_delta, std::memory_order_relaxed);
  std::size_t new_current = old_current + size_delta;

  std::size_t old_hwm = highwatermark_.load(std::memory_order_relaxed);
  while (new_current > old_hwm) {
    if (highwatermark_.compare_exchange_weak(old_hwm, new_current, std::memory_order_relaxed)) {
      break;
    }
  }
}

void memory::update_actual_size(std::ptrdiff_t size_delta)
{
  actual_size_.fetch_add(size_delta, std::memory_order_relaxed);
}

} // namespace umpire
