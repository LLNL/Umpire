//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/detail/registry.hpp"
#include "umpire/memory.hpp"

#include <algorithm>
#include <cstdint>

namespace umpire {
namespace detail {

registry::registry() = default;
registry::~registry() = default;

registry& registry::get()
{
  static registry instance;
  return instance;
}

int registry::get_id()
{
  return next_id_.fetch_add(1, std::memory_order_relaxed);
}

void registry::register_allocator(memory* alloc)
{
  std::lock_guard<std::mutex> lock{allocator_mutex_};
  allocator_list_.push_back(alloc);
  allocator_by_name_[alloc->get_name()] = alloc;
  allocator_by_id_[alloc->get_id()] = alloc;
}

void registry::deregister_allocator(memory* alloc)
{
  std::lock_guard<std::mutex> lock{allocator_mutex_};
  allocator_list_.erase(std::remove(allocator_list_.begin(), allocator_list_.end(), alloc), allocator_list_.end());

  for (auto it = allocator_by_name_.begin(); it != allocator_by_name_.end();) {
    if (it->second == alloc) {
      it = allocator_by_name_.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = allocator_by_id_.begin(); it != allocator_by_id_.end();) {
    if (it->second == alloc) {
      it = allocator_by_id_.erase(it);
    } else {
      ++it;
    }
  }
}

memory* registry::find_allocator_by_id(int id)
{
  std::lock_guard<std::mutex> lock{allocator_mutex_};
  auto it = allocator_by_id_.find(id);
  return (it == allocator_by_id_.end()) ? nullptr : it->second;
}

memory* registry::find_allocator_by_name(const std::string& name)
{
  std::lock_guard<std::mutex> lock{allocator_mutex_};
  auto it = allocator_by_name_.find(name);
  return (it == allocator_by_name_.end()) ? nullptr : it->second;
}

std::vector<memory*> registry::get_allocators()
{
  std::lock_guard<std::mutex> lock{allocator_mutex_};
  return allocator_list_;
}

void registry::register_allocation(const allocation_record& record)
{
  std::lock_guard<std::mutex> lock{allocation_mutex_};
  allocation_map_[record.ptr] = record;
}

std::optional<allocation_record> registry::find_allocation(void* ptr) const
{
  std::lock_guard<std::mutex> lock{allocation_mutex_};
  auto it = allocation_map_.find(ptr);
  return (it == allocation_map_.end()) ? std::nullopt : std::optional<allocation_record>{it->second};
}

std::optional<allocation_record> registry::find_containing_allocation(void* ptr) const
{
  std::lock_guard<std::mutex> lock{allocation_mutex_};

  // The candidate record is the one with the greatest base pointer <= ptr.
  auto it = allocation_map_.upper_bound(ptr);
  if (it == allocation_map_.begin()) {
    return std::nullopt;
  }
  --it;

  const auto target = reinterpret_cast<std::uintptr_t>(ptr);
  const auto begin = reinterpret_cast<std::uintptr_t>(it->second.ptr);
  if (target >= begin && (target - begin) < it->second.size) {
    return it->second;
  }

  return std::nullopt;
}

void registry::remove_allocation(void* ptr)
{
  std::lock_guard<std::mutex> lock{allocation_mutex_};
  allocation_map_.erase(ptr);
}

bool registry::has_allocation(void* ptr) const
{
  std::lock_guard<std::mutex> lock{allocation_mutex_};
  return allocation_map_.find(ptr) != allocation_map_.end();
}

std::vector<allocation_record> registry::find_allocations_by_memory(const memory* mem) const
{
  std::lock_guard<std::mutex> lock{allocation_mutex_};
  std::vector<allocation_record> records;

  for (const auto& [base, record] : allocation_map_) {
    (void)base;
    if (record.strategy == mem) {
      records.push_back(record);
    }
  }

  return records;
}

std::vector<allocation_record> registry::find_allocations_by_memory(int id) const
{
  std::lock_guard<std::mutex> lock{allocation_mutex_};
  std::vector<allocation_record> records;

  for (const auto& [base, record] : allocation_map_) {
    (void)base;
    if (record.strategy && record.strategy->get_id() == id) {
      records.push_back(record);
    }
  }

  return records;
}

} // namespace detail
} // namespace umpire
