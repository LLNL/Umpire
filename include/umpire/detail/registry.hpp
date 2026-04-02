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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "umpire/allocation_record.hpp"

namespace umpire {
class memory;
}

namespace umpire {
namespace detail {

/*!
 * \brief Process-wide registry for API v2 allocator and allocation metadata.
 *
 * Thread safety guarantees:
 * - `get_id()`, allocator registration, allocator lookup, and allocation-map
 *   membership queries are synchronized internally and may be called
 *   concurrently.
 * - `find_allocation()` returns a copy of the allocation metadata, so callers
 *   never retain references into internal registry storage across concurrent
 *   mutations.
 */
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
  //! Access the process-wide singleton registry.
  static registry& get();

  //! Generate a unique integer identifier for a memory object.
  int get_id();

  //! Register a memory object for lookup by id and name.
  void register_allocator(memory* alloc);
  //! Remove a memory object from registry lookup tables.
  void deregister_allocator(memory* alloc);

  //! Find a registered memory object by id.
  memory* find_allocator_by_id(int id);
  //! Find a registered memory object by name.
  memory* find_allocator_by_name(const std::string& name);
  //! Return a snapshot of registered memory objects.
  std::vector<memory*> get_allocators();

  //! Register a live tracked allocation.
  void register_allocation(const allocation_record& record);
  //! Find the record whose base pointer matches `ptr`.
  std::optional<allocation_record> find_allocation(void* ptr) const;
  //! Find the record whose range contains `ptr`.
  std::optional<allocation_record> find_containing_allocation(void* ptr) const;
  //! Remove the record whose base pointer matches `ptr`.
  void remove_allocation(void* ptr);
  //! Return whether `ptr` is the base pointer of a tracked allocation.
  bool has_allocation(void* ptr) const;
};

} // namespace detail
} // namespace umpire

#endif // UMPIRE_detail_registry_HPP
