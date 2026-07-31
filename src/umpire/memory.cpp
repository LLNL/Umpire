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

bool should_bridge_to_v1_host_allocator(const memory& mem)
{
  return mem.get_name() == "HOST" && mem.get_platform() == resource::Platform::host;
}

void register_with_v1_host_allocator(void* ptr, std::size_t size)
{
  auto& rm = ResourceManager::getInstance();
  if (!rm.hasAllocator(ptr)) {
    auto host_allocator = rm.getAllocator("HOST");
    rm.registerAllocation(ptr, util::AllocationRecord{ptr, size, host_allocator.getAllocationStrategy()});
  }
}

void deregister_from_v1_host_allocator(void* ptr)
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
  if (should_bridge_to_v1_host_allocator(*this)) {
    register_with_v1_host_allocator(ptr, size);
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
  if (should_bridge_to_v1_host_allocator(*this)) {
    deregister_from_v1_host_allocator(ptr);
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
