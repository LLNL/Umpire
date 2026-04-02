//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_memory_HPP
#define UMPIRE_memory_HPP

#include <atomic>
#include <cstddef>
#include <string>

#include "umpire/resource/platform.hpp"

namespace umpire {
struct allocation_record;
}

namespace umpire {

/*!
 * \brief Common API v2 base class for memory resources and strategies.
 *
 * Thread safety guarantees:
 * - Construction/destruction register with the shared v2 registry.
 * - Read-only introspection (`get_id()`, `get_name()`, `get_current_size()`,
 *   `get_actual_size()`, `get_highwatermark()`) is safe for concurrent reads
 *   after construction.
 * - Thread safety of `allocate()` and `deallocate()` is defined by the
 *   concrete implementation. Use wrappers such as `thread_safe<Memory>` when
 *   concurrent mutation must be serialized.
 */
class memory {
private:
  int id_;
  std::string name_;

  std::atomic<std::size_t> current_size_{0};
  std::atomic<std::size_t> actual_size_{0};
  std::atomic<std::size_t> highwatermark_{0};

protected:
  void track_allocation(void* ptr, std::size_t size);
  void untrack_allocation(void* ptr);

  void update_statistics(std::ptrdiff_t size_delta);

public:
  explicit memory(const std::string& name);
  virtual ~memory();

  memory(const memory&) = delete;
  memory& operator=(const memory&) = delete;
  memory(memory&&) = delete;
  memory& operator=(memory&&) = delete;

  virtual void* allocate(std::size_t size) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual resource::Platform get_platform() const = 0;

  int get_id() const { return id_; }
  const std::string& get_name() const { return name_; }
  std::size_t get_current_size() const { return current_size_.load(std::memory_order_relaxed); }
  std::size_t get_actual_size() const { return actual_size_.load(std::memory_order_relaxed); }
  std::size_t get_highwatermark() const { return highwatermark_.load(std::memory_order_relaxed); }
};

} // namespace umpire

#endif // UMPIRE_memory_HPP
