//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_recorder_chain_HPP
#define UMPIRE_recorder_chain_HPP

#include "umpire/event/event_store.hpp"

#include <vector>

namespace umpire {
namespace event {

struct event;
struct allocate;
struct named_allocate;
struct allocate_resource;
struct deallocate;
struct deallocate_resource;

/**
 * @brief Chain of event stores that records events to multiple stores
 *
 * This class allows recording events to multiple event_store implementations
 * simultaneously (e.g., json_file_store for replay + streaming_event_sink for metrics).
 *
 * Thread safety: If multiple threads call record() concurrently, all underlying
 * stores must be thread-safe.
 */
class recorder_chain : public event_store {
 public:
  recorder_chain() = default;

  /**
   * @brief Add an event store to the chain
   * @param store Pointer to event store (non-owning, must outlive recorder_chain)
   */
  void add_store(event_store* store);

  /**
   * @brief Record an event to all stores in the chain
   *
   * If one store throws an exception, it is caught and logged, but other
   * stores continue to receive the event (error isolation).
   */
  void insert(const event& e) override;
  void insert(const allocate& e) override;
  void insert(const named_allocate& e) override;
  void insert(const allocate_resource& e) override;
  void insert(const deallocate& e) override;
  void insert(const deallocate_resource& e) override;

  std::vector<event> get_events() override;

 private:
  std::vector<event_store*> m_stores;  // Non-owning pointers
};

} // namespace event
} // namespace umpire

#endif // UMPIRE_recorder_chain_HPP
