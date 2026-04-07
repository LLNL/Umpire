//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_prometheus_backend_HPP
#define UMPIRE_prometheus_backend_HPP

#include "umpire/event/metrics_aggregator.hpp"
#include "umpire/event/streaming_backend.hpp"

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

namespace umpire {
namespace event {

/**
 * @brief Prometheus remote-write backend for streaming metrics
 *
 * This backend aggregates allocation/deallocation events into Prometheus metrics
 * and periodically sends them via HTTP POST to a Prometheus remote-write endpoint.
 *
 * Architecture:
 * - Events are aggregated in-process using metrics_aggregator (atomic counters)
 * - Background thread flushes metrics every N seconds (default: 30s)
 * - Uses Prometheus remote-write protocol (protobuf + snappy compression)
 * - Retry logic: 3 attempts with exponential backoff, then permanent fallback
 *
 * Thread safety: All methods are thread-safe. send_batch() updates atomics,
 * flush_loop() runs in background thread.
 */
class prometheus_backend : public streaming_backend {
 public:
  /**
   * @brief Construct Prometheus backend
   * @param endpoint Prometheus remote-write endpoint (e.g., "http://localhost:9090/api/v1/write")
   * @param rank MPI rank or process ID for labeling
   * @param job_name Job name label (default: "umpire")
   * @param environment Environment label (default: "production")
   * @param flush_interval How often to flush metrics (default: 30 seconds)
   */
  prometheus_backend(const std::string& endpoint, int rank = 0, const std::string& job_name = "umpire",
                     const std::string& environment = "production",
                     std::chrono::seconds flush_interval = std::chrono::seconds(30));

  ~prometheus_backend();

  // Implement streaming_backend interface
  void send_batch(std::span<const event> events) override;
  void send_batch(std::span<const allocate> events) override;
  void send_batch(std::span<const named_allocate> events) override;
  void send_batch(std::span<const allocate_resource> events) override;
  void send_batch(std::span<const deallocate> events) override;
  void send_batch(std::span<const deallocate_resource> events) override;

  bool is_healthy() const override;

  // Accessors for testing
  const metrics_aggregator& get_aggregator() const { return m_aggregator; }

 private:
  metrics_aggregator m_aggregator;
  std::string m_remote_write_endpoint;
  int m_rank;
  std::string m_job_name;
  std::string m_environment;
  std::chrono::seconds m_flush_interval;

  // Background flush thread
  std::thread m_flush_thread;
  std::atomic<bool> m_running{true};

  // Health tracking
  enum class State { ACTIVE, RETRYING, FALLBACK };
  std::atomic<State> m_state{State::ACTIVE};
  std::atomic<std::uint64_t> m_send_failures{0};

  /**
   * @brief Background thread that periodically flushes metrics
   */
  void flush_loop();

  /**
   * @brief Send current metrics to Prometheus via HTTP POST
   * @return true if successful, false otherwise
   */
  bool send_metrics_to_prometheus();

  /**
   * @brief Send metrics with retry logic (3 attempts, exponential backoff)
   * @return true if successful, false if all retries failed
   */
  bool send_with_retry();
};

} // namespace event
} // namespace umpire

#endif // UMPIRE_prometheus_backend_HPP
