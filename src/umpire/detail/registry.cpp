//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/detail/registry.hpp"
#include "umpire/memory.hpp"

#include <algorithm>

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

allocation_record* registry::find_allocation(void* ptr)
{
  std::lock_guard<std::mutex> lock{allocation_mutex_};
  auto it = allocation_map_.find(ptr);
  return (it == allocation_map_.end()) ? nullptr : &it->second;
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

} // namespace detail
} // namespace umpire
