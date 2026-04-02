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
#include "umpire/util/AllocationRecord.hpp"

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
}

void memory::untrack_allocation(void* ptr)
{
  auto record = detail::registry::get().find_allocation(ptr);
  if (!record) {
    throw unknown_allocation(fmt::format("Attempted to deallocate unknown pointer {:p}", ptr));
  }

  std::size_t size = record->size;
  detail::registry::get().remove_allocation(ptr);
  if (should_bridge_to_v1_host_allocator(*this)) {
    deregister_from_v1_host_allocator(ptr);
  }
  update_statistics(-static_cast<std::ptrdiff_t>(size));
}

void memory::update_statistics(std::ptrdiff_t size_delta)
{
  // Update current size
  std::size_t old_current = current_size_.fetch_add(size_delta, std::memory_order_relaxed);
  std::size_t new_current = old_current + size_delta;

  // Update highwatermark if needed
  std::size_t old_hwm = highwatermark_.load(std::memory_order_relaxed);
  while (new_current > old_hwm) {
    if (highwatermark_.compare_exchange_weak(old_hwm, new_current, std::memory_order_relaxed)) {
      break;
    }
  }

  // Update actual size (always same as current for base implementation)
  actual_size_.fetch_add(size_delta, std::memory_order_relaxed);
}

} // namespace umpire
