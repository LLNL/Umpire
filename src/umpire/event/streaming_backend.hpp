//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_streaming_backend_HPP
#define UMPIRE_streaming_backend_HPP

#include <span>

namespace umpire {
namespace event {

struct event;
struct allocate;
struct named_allocate;
struct allocate_resource;
struct deallocate;
struct deallocate_resource;

/**
 * @brief Abstract interface for streaming backends that receive batched events
 *
 * Implementations of this interface handle sending events to external monitoring
 * systems (e.g., Prometheus, InfluxDB, OpenTelemetry). The streaming_event_sink
 * batches events and calls send_batch() methods on the backend.
 *
 * Thread safety: Implementations must be thread-safe for send_batch() calls,
 * as they may be called from the background export thread.
 */
class streaming_backend {
 public:
  virtual ~streaming_backend() = default;

  /**
   * @brief Send a batch of generic events
   * @param events Span of events to send
   */
  virtual void send_batch(std::span<const event> events) = 0;

  /**
   * @brief Send a batch of allocate events
   * @param events Span of allocate events to send
   */
  virtual void send_batch(std::span<const allocate> events) = 0;

  /**
   * @brief Send a batch of named_allocate events
   * @param events Span of named_allocate events to send
   */
  virtual void send_batch(std::span<const named_allocate> events) = 0;

  /**
   * @brief Send a batch of allocate_resource events
   * @param events Span of allocate_resource events to send
   */
  virtual void send_batch(std::span<const allocate_resource> events) = 0;

  /**
   * @brief Send a batch of deallocate events
   * @param events Span of deallocate events to send
   */
  virtual void send_batch(std::span<const deallocate> events) = 0;

  /**
   * @brief Send a batch of deallocate_resource events
   * @param events Span of deallocate_resource events to send
   */
  virtual void send_batch(std::span<const deallocate_resource> events) = 0;

  /**
   * @brief Check if the backend is healthy and able to send events
   * @return true if backend is operational, false otherwise
   */
  virtual bool is_healthy() const = 0;
};

} // namespace event
} // namespace umpire

#endif // UMPIRE_streaming_backend_HPP
