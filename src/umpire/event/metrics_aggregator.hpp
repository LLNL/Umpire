//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_metrics_aggregator_HPP
#define UMPIRE_metrics_aggregator_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace umpire {
namespace event {

/**
 * @brief Aggregates allocation/deallocation events into Prometheus-style metrics
 *
 * This class maintains atomic counters and gauges for tracking memory usage,
 * allocation rates, and size distributions. Metrics are aggregated in-process
 * and can be exported in Prometheus text or remote-write format.
 *
 * Thread safety: All methods are thread-safe. Counters/gauges use atomics,
 * per-allocator tracking uses mutex.
 */
class metrics_aggregator {
 public:
  metrics_aggregator();

  /**
   * @brief Record an allocation event
   * @param ptr Pointer to allocated memory
   * @param size Size of allocation in bytes
   * @param allocator_ref Pointer to allocator (used as identifier)
   */
  void record_allocate(void* ptr, std::size_t size, void* allocator_ref);

  /**
   * @brief Record a deallocation event
   * @param ptr Pointer to deallocated memory (used to lookup original size)
   * @param allocator_ref Pointer to allocator (used as identifier)
   */
  void record_deallocate(void* ptr, void* allocator_ref);

  /**
   * @brief Render metrics in Prometheus text exposition format
   * @return String containing metrics in Prometheus text format
   */
  std::string render_prometheus_text() const;

  /**
   * @brief Render metrics in Prometheus remote-write protobuf format
   * @param rank MPI rank (or 0 for non-MPI)
   * @param job_name Job name label (optional)
   * @param environment Environment label (optional, e.g., "production")
   * @return Serialized protobuf bytes
   */
  std::vector<std::uint8_t> render_remote_write(int rank = 0, const std::string& job_name = "",
                                                 const std::string& environment = "") const;

  // Accessors for testing
  std::uint64_t allocations_total() const { return m_allocations_total.load(std::memory_order_relaxed); }
  std::uint64_t deallocations_total() const { return m_deallocations_total.load(std::memory_order_relaxed); }
  std::int64_t bytes_allocated() const { return m_bytes_allocated.load(std::memory_order_relaxed); }

 private:
  // Global metrics (atomic for thread safety)
  std::atomic<std::uint64_t> m_allocations_total{0};
  std::atomic<std::uint64_t> m_deallocations_total{0};
  std::atomic<std::int64_t> m_bytes_allocated{0};  // Signed for gauge (can go negative)

  // Histogram buckets for allocation sizes (exponential: 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, +Inf)
  // Index 0 = <1KB, 1 = <4KB, ..., 8 = +Inf
  std::array<std::atomic<std::uint64_t>, 9> m_size_histogram{};
  std::atomic<std::uint64_t> m_size_sum{0};  // Sum of all allocation sizes (for avg calculation)

  // Per-allocator metrics (mutex-protected)
  struct PerAllocatorMetrics {
    std::uint64_t allocations{0};
    std::uint64_t deallocations{0};
    std::int64_t bytes{0};
  };
  mutable std::mutex m_allocator_mutex;
  std::unordered_map<std::string, PerAllocatorMetrics> m_per_allocator;

  // Track allocation sizes by pointer (mutex-protected)
  // Key: ptr, Value: (size, allocator_ref_string)
  std::unordered_map<void*, std::pair<std::size_t, std::string>> m_allocation_map;

  /**
   * @brief Get histogram bucket index for a given size
   * @param size Allocation size in bytes
   * @return Bucket index (0-8)
   */
  static std::size_t get_histogram_bucket(std::size_t size);

  /**
   * @brief Get histogram bucket upper bound in bytes
   * @param bucket_index Bucket index (0-8)
   * @return Upper bound in bytes (+Inf for last bucket)
   */
  static std::size_t get_bucket_upper_bound(std::size_t bucket_index);

  /**
   * @brief Convert allocator ref pointer to string identifier
   * @param allocator_ref Pointer to allocator
   * @return String representation of allocator (e.g., "0x7f8b3c000000")
   */
  static std::string allocator_ref_to_string(void* allocator_ref);
};

} // namespace event
} // namespace umpire

#endif // UMPIRE_metrics_aggregator_HPP
