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
  /*!
   * \brief Register a live allocation in the shared API v2 registry.
   *
   * \param ptr Allocation base pointer.
   * \param size Requested size in bytes.
   */
  void track_allocation(void* ptr, std::size_t size);

  /*!
   * \brief Remove a live allocation from the shared API v2 registry.
   *
   * \param ptr Allocation base pointer.
   */
  void untrack_allocation(void* ptr);

  /*!
   * \brief Update current and peak statistics by a signed byte delta.
   *
   * Applies the delta to both the live-byte and backend-accounted counters;
   * appropriate for resources where the two always move together.
   *
   * \param size_delta Signed change in live bytes.
   */
  void update_statistics(std::ptrdiff_t size_delta);

  /*!
   * \brief Update live-byte statistics (and peak watermark) only.
   *
   * Pooling strategies use this for user-facing allocations, tracking the
   * backing memory separately via update_actual_size().
   *
   * \param size_delta Signed change in live bytes.
   */
  void update_current_size(std::ptrdiff_t size_delta);

  /*!
   * \brief Update backend-accounted bytes only.
   *
   * \param size_delta Signed change in backend-accounted bytes.
   */
  void update_actual_size(std::ptrdiff_t size_delta);

public:
  /*!
   * \brief Construct a named memory object.
   *
   * Construction also registers the object with the shared v2 registry.
   *
   * \param name Human-readable name for diagnostics and lookup.
   */
  explicit memory(const std::string& name);

  //! Deregister this object from the shared registry.
  virtual ~memory();

  memory(const memory&) = delete;
  memory& operator=(const memory&) = delete;
  memory(memory&&) = delete;
  memory& operator=(memory&&) = delete;

  /*!
   * \brief Allocate raw storage.
   *
   * \param size Number of bytes to allocate.
   * \return Pointer to the allocated storage.
   */
  virtual void* allocate(std::size_t size) = 0;

  /*!
   * \brief Release storage previously obtained from allocate().
   *
   * \param ptr Pointer to the allocation to release.
   */
  virtual void deallocate(void* ptr) = 0;

  /*!
   * \brief Report the runtime platform of this memory object.
   *
   * \return Runtime platform tag matching the underlying resource.
   */
  virtual resource::Platform get_platform() const = 0;

  //! Stable registry-assigned identifier.
  int get_id() const { return id_; }
  //! Human-readable name supplied at construction.
  const std::string& get_name() const { return name_; }
  //! Current live bytes attributed to this object.
  std::size_t get_current_size() const { return current_size_.load(std::memory_order_relaxed); }
  //! Current backend-accounted bytes attributed to this object.
  std::size_t get_actual_size() const { return actual_size_.load(std::memory_order_relaxed); }
  //! Peak live-byte watermark observed for this object.
  std::size_t get_highwatermark() const { return highwatermark_.load(std::memory_order_relaxed); }
};

} // namespace umpire

#endif // UMPIRE_memory_HPP
