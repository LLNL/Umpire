//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/streaming_event_sink.hpp"

#include "umpire/event/event.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

#include <thread>
#include <vector>

namespace umpire {
namespace event {

streaming_event_sink::streaming_event_sink(streaming_backend* backend, std::size_t max_queue_size)
    : m_backend(backend), m_max_size(max_queue_size)
{
  if (m_backend == nullptr) {
    UMPIRE_ERROR(umpire::runtime_error, "streaming_event_sink: backend cannot be null");
  }

  // Start background export thread
  m_export_thread = std::thread(&streaming_event_sink::export_loop, this);
}

streaming_event_sink::~streaming_event_sink()
{
  // Signal shutdown
  m_running.store(false, std::memory_order_release);

  // Wake up the export thread
  m_cv.notify_one();

  // Wait for thread to finish
  if (m_export_thread.joinable()) {
    m_export_thread.join();
  }
}

template <typename T>
bool streaming_event_sink::try_push_with_backpressure(const T& e)
{
  // Attempt immediate push
  {
    std::unique_lock<std::mutex> lock(m_mutex);

    // Check queue size based on event type
    std::size_t total_size = m_event_queue.size() + m_allocate_queue.size() + m_named_allocate_queue.size() +
                             m_allocate_resource_queue.size() + m_deallocate_queue.size() +
                             m_deallocate_resource_queue.size();

    if (total_size < m_max_size) {
      // Queue not full - push immediately
      if constexpr (std::is_same_v<T, event>) {
        m_event_queue.push(e);
      } else if constexpr (std::is_same_v<T, allocate>) {
        m_allocate_queue.push(e);
      } else if constexpr (std::is_same_v<T, named_allocate>) {
        m_named_allocate_queue.push(e);
      } else if constexpr (std::is_same_v<T, allocate_resource>) {
        m_allocate_resource_queue.push(e);
      } else if constexpr (std::is_same_v<T, deallocate>) {
        m_deallocate_queue.push(e);
      } else if constexpr (std::is_same_v<T, deallocate_resource>) {
        m_deallocate_resource_queue.push(e);
      }

      m_cv.notify_one();
      return true;
    }
  }

  // Queue full - apply backpressure (retry with exponential backoff)
  for (int attempt = 0; attempt < 3; ++attempt) {
    // Sleep with exponential backoff: 100μs, 200μs, 300μs
    std::this_thread::sleep_for(std::chrono::microseconds(100 * (attempt + 1)));

    std::unique_lock<std::mutex> lock(m_mutex);
    std::size_t total_size = m_event_queue.size() + m_allocate_queue.size() + m_named_allocate_queue.size() +
                             m_allocate_resource_queue.size() + m_deallocate_queue.size() +
                             m_deallocate_resource_queue.size();

    if (total_size < m_max_size) {
      if constexpr (std::is_same_v<T, event>) {
        m_event_queue.push(e);
      } else if constexpr (std::is_same_v<T, allocate>) {
        m_allocate_queue.push(e);
      } else if constexpr (std::is_same_v<T, named_allocate>) {
        m_named_allocate_queue.push(e);
      } else if constexpr (std::is_same_v<T, allocate_resource>) {
        m_allocate_resource_queue.push(e);
      } else if constexpr (std::is_same_v<T, deallocate>) {
        m_deallocate_queue.push(e);
      } else if constexpr (std::is_same_v<T, deallocate_resource>) {
        m_deallocate_resource_queue.push(e);
      }

      m_cv.notify_one();
      return true;
    }
  }

  // Still full after backpressure - drop event
  return false;
}

void streaming_event_sink::insert(const event& e)
{
  if (!try_push_with_backpressure(e)) {
    m_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void streaming_event_sink::insert(const allocate& e)
{
  if (!try_push_with_backpressure(e)) {
    m_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void streaming_event_sink::insert(const named_allocate& e)
{
  if (!try_push_with_backpressure(e)) {
    m_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void streaming_event_sink::insert(const allocate_resource& e)
{
  if (!try_push_with_backpressure(e)) {
    m_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void streaming_event_sink::insert(const deallocate& e)
{
  if (!try_push_with_backpressure(e)) {
    m_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void streaming_event_sink::insert(const deallocate_resource& e)
{
  if (!try_push_with_backpressure(e)) {
    m_overflow_count.fetch_add(1, std::memory_order_relaxed);
  }
}

std::vector<event> streaming_event_sink::get_events()
{
  // Streaming sink doesn't support reading events back
  return {};
}

template <typename T>
bool streaming_event_sink::send_batch_with_retry(const std::vector<T>& events)
{
  if (events.empty()) {
    return true;
  }

  if (m_state.load(std::memory_order_acquire) == State::FALLBACK) {
    // Permanent fallback - don't attempt send
    return false;
  }

  if (m_state.load(std::memory_order_acquire) == State::RETRYING) {
    // Check if it's time to retry
    if (std::chrono::steady_clock::now() < m_next_retry_time) {
      return false;  // Not time yet
    }
  }

  try {
    m_backend->send_batch(std::span<const T>(events.data(), events.size()));

    // Success - reset failure count
    if (m_consecutive_failures > 0) {
      UMPIRE_LOG(Info, "Streaming backend recovered after failures");
      m_consecutive_failures = 0;
      m_state.store(State::ACTIVE, std::memory_order_release);
    }
    return true;

  } catch (const std::exception& ex) {
    m_consecutive_failures++;

    if (m_consecutive_failures <= MAX_RETRIES) {
      // Exponential backoff: 1s, 2s, 4s
      auto backoff = std::chrono::seconds(1 << (m_consecutive_failures - 1));
      m_next_retry_time = std::chrono::steady_clock::now() + backoff;
      m_state.store(State::RETRYING, std::memory_order_release);

      UMPIRE_LOG(Warning,
                 "Streaming backend failed (attempt " << m_consecutive_failures << "/" << MAX_RETRIES
                                                       << "), retrying in " << backoff.count() << "s: " << ex.what());
    } else {
      // Permanent fallback
      m_state.store(State::FALLBACK, std::memory_order_release);
      UMPIRE_LOG(Error, "Streaming backend failed after " << MAX_RETRIES
                                                           << " retries. Disabling streaming permanently.");
    }
    return false;
  }
}

void streaming_event_sink::export_loop()
{
  std::vector<event> event_batch;
  std::vector<allocate> allocate_batch;
  std::vector<named_allocate> named_allocate_batch;
  std::vector<allocate_resource> allocate_resource_batch;
  std::vector<deallocate> deallocate_batch;
  std::vector<deallocate_resource> deallocate_resource_batch;

  event_batch.reserve(1000);
  allocate_batch.reserve(1000);
  named_allocate_batch.reserve(1000);
  allocate_resource_batch.reserve(1000);
  deallocate_batch.reserve(1000);
  deallocate_resource_batch.reserve(1000);

  while (m_running.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lock(m_mutex);

      // Wait for events or timeout (100ms)
      m_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
        return !m_event_queue.empty() || !m_allocate_queue.empty() || !m_named_allocate_queue.empty() ||
               !m_allocate_resource_queue.empty() || !m_deallocate_queue.empty() ||
               !m_deallocate_resource_queue.empty() || !m_running.load(std::memory_order_acquire);
      });

      // Pop up to 1000 events of each type
      while (!m_event_queue.empty() && event_batch.size() < 1000) {
        event_batch.push_back(std::move(m_event_queue.front()));
        m_event_queue.pop();
      }

      while (!m_allocate_queue.empty() && allocate_batch.size() < 1000) {
        allocate_batch.push_back(std::move(m_allocate_queue.front()));
        m_allocate_queue.pop();
      }

      while (!m_named_allocate_queue.empty() && named_allocate_batch.size() < 1000) {
        named_allocate_batch.push_back(std::move(m_named_allocate_queue.front()));
        m_named_allocate_queue.pop();
      }

      while (!m_allocate_resource_queue.empty() && allocate_resource_batch.size() < 1000) {
        allocate_resource_batch.push_back(std::move(m_allocate_resource_queue.front()));
        m_allocate_resource_queue.pop();
      }

      while (!m_deallocate_queue.empty() && deallocate_batch.size() < 1000) {
        deallocate_batch.push_back(std::move(m_deallocate_queue.front()));
        m_deallocate_queue.pop();
      }

      while (!m_deallocate_resource_queue.empty() && deallocate_resource_batch.size() < 1000) {
        deallocate_resource_batch.push_back(std::move(m_deallocate_resource_queue.front()));
        m_deallocate_resource_queue.pop();
      }
    }  // Release lock while sending

    // Send batches (outside lock)
    send_batch_with_retry(event_batch);
    send_batch_with_retry(allocate_batch);
    send_batch_with_retry(named_allocate_batch);
    send_batch_with_retry(allocate_resource_batch);
    send_batch_with_retry(deallocate_batch);
    send_batch_with_retry(deallocate_resource_batch);

    // Clear batches for next iteration
    event_batch.clear();
    allocate_batch.clear();
    named_allocate_batch.clear();
    allocate_resource_batch.clear();
    deallocate_batch.clear();
    deallocate_resource_batch.clear();
  }

  // Final flush on shutdown
  {
    std::unique_lock<std::mutex> lock(m_mutex);

    while (!m_event_queue.empty()) {
      event_batch.push_back(std::move(m_event_queue.front()));
      m_event_queue.pop();
    }

    while (!m_allocate_queue.empty()) {
      allocate_batch.push_back(std::move(m_allocate_queue.front()));
      m_allocate_queue.pop();
    }

    while (!m_named_allocate_queue.empty()) {
      named_allocate_batch.push_back(std::move(m_named_allocate_queue.front()));
      m_named_allocate_queue.pop();
    }

    while (!m_allocate_resource_queue.empty()) {
      allocate_resource_batch.push_back(std::move(m_allocate_resource_queue.front()));
      m_allocate_resource_queue.pop();
    }

    while (!m_deallocate_queue.empty()) {
      deallocate_batch.push_back(std::move(m_deallocate_queue.front()));
      m_deallocate_queue.pop();
    }

    while (!m_deallocate_resource_queue.empty()) {
      deallocate_resource_batch.push_back(std::move(m_deallocate_resource_queue.front()));
      m_deallocate_resource_queue.pop();
    }
  }

  // Final send
  send_batch_with_retry(event_batch);
  send_batch_with_retry(allocate_batch);
  send_batch_with_retry(named_allocate_batch);
  send_batch_with_retry(allocate_resource_batch);
  send_batch_with_retry(deallocate_batch);
  send_batch_with_retry(deallocate_resource_batch);
}

} // namespace event
} // namespace umpire
