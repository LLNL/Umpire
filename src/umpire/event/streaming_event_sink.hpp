//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_streaming_event_sink_HPP
#define UMPIRE_streaming_event_sink_HPP

#include "umpire/event/event.hpp"
#include "umpire/event/event_store.hpp"
#include "umpire/event/streaming_backend.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace umpire {
namespace event {

/**
 * @brief Event store that streams events to a backend via a background thread
 *
 * This class implements a thread-safe queue (using std::mutex) that buffers
 * events from allocation threads (2-4 threads typically) and sends them in
 * batches to a streaming backend (e.g., Prometheus) via a background thread.
 *
 * Thread safety: Multiple threads can call insert() concurrently. A dedicated
 * background thread pops events and sends batches to the backend.
 *
 * Backpressure: If the queue is full, insert() will briefly retry with exponential
 * backoff before dropping the event. Dropped events are tracked via overflow counter.
 */
class streaming_event_sink : public event_store {
 public:
  /**
   * @brief Construct a streaming event sink
   * @param backend Pointer to backend implementation (non-owning)
   * @param max_queue_size Maximum number of events to buffer (default: 16384)
   */
  explicit streaming_event_sink(streaming_backend* backend, std::size_t max_queue_size = 16384);

  /**
   * @brief Destructor - flushes remaining events and joins background thread
   */
  ~streaming_event_sink();

  // Delete copy/move operations (manages thread and mutex)
  streaming_event_sink(const streaming_event_sink&) = delete;
  streaming_event_sink& operator=(const streaming_event_sink&) = delete;
  streaming_event_sink(streaming_event_sink&&) = delete;
  streaming_event_sink& operator=(streaming_event_sink&&) = delete;

  void insert(const event& e) override;
  void insert(const allocate& e) override;
  void insert(const named_allocate& e) override;
  void insert(const allocate_resource& e) override;
  void insert(const deallocate& e) override;
  void insert(const deallocate_resource& e) override;

  std::vector<event> get_events() override;

  /**
   * @brief Get the number of events dropped due to queue overflow
   * @return Count of dropped events since construction
   */
  std::uint64_t get_overflow_count() const { return m_overflow_count.load(std::memory_order_relaxed); }

 private:
  /**
   * @brief Try to push an event with backpressure (block-then-drop)
   * @param e Event to push
   * @return true if successfully queued, false if dropped
   */
  template <typename T>
  bool try_push_with_backpressure(const T& e);

  /**
   * @brief Background thread function - pops events and sends batches
   */
  void export_loop();

  /**
   * @brief Send a batch of events to the backend with retry logic
   * @param events Vector of events to send
   * @return true if successful, false if permanent failure
   */
  template <typename T>
  bool send_batch_with_retry(const std::vector<T>& events);

  streaming_backend* m_backend;  // Non-owning pointer
  const std::size_t m_max_size;

  // Thread-safe queue (mutex-protected)
  std::mutex m_mutex;
  std::condition_variable m_cv;

  // Separate queues for each event type (simpler than std::variant)
  std::queue<event> m_event_queue;
  std::queue<allocate> m_allocate_queue;
  std::queue<named_allocate> m_named_allocate_queue;
  std::queue<allocate_resource> m_allocate_resource_queue;
  std::queue<deallocate> m_deallocate_queue;
  std::queue<deallocate_resource> m_deallocate_resource_queue;

  std::atomic<std::uint64_t> m_overflow_count{0};

  // Background thread
  std::thread m_export_thread;
  std::atomic<bool> m_running{true};

  // Retry state
  enum class State { ACTIVE, RETRYING, FALLBACK };
  std::atomic<State> m_state{State::ACTIVE};
  int m_consecutive_failures{0};
  static constexpr int MAX_RETRIES = 3;
  std::chrono::steady_clock::time_point m_next_retry_time;
};

} // namespace event
} // namespace umpire

#endif // UMPIRE_streaming_event_sink_HPP
