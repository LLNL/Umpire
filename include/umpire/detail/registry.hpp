//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_detail_registry_HPP
#define UMPIRE_detail_registry_HPP

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "umpire/allocation_record.hpp"

namespace umpire {
class memory;
}

namespace umpire {
namespace detail {

class registry {
private:
  registry();
  ~registry();

  registry(const registry&) = delete;
  registry& operator=(const registry&) = delete;

  std::atomic<int> next_id_{0};

  std::vector<memory*> allocator_list_;
  std::unordered_map<std::string, memory*> allocator_by_name_;
  std::unordered_map<int, memory*> allocator_by_id_;

  std::unordered_map<void*, allocation_record> allocation_map_;

  mutable std::mutex allocator_mutex_;
  mutable std::mutex allocation_mutex_;

public:
  static registry& get();

  int get_id();

  void register_allocator(memory* alloc);
  void deregister_allocator(memory* alloc);

  memory* find_allocator_by_id(int id);
  memory* find_allocator_by_name(const std::string& name);
  std::vector<memory*> get_allocators();

  void register_allocation(const allocation_record& record);
  allocation_record* find_allocation(void* ptr);
  void remove_allocation(void* ptr);
  bool has_allocation(void* ptr) const;
};

} // namespace detail
} // namespace umpire

#endif // UMPIRE_detail_registry_HPP
