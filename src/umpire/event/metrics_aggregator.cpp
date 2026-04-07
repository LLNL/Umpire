//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/event/metrics_aggregator.hpp"

#include <chrono>
#include <cmath>
#include <sstream>

namespace umpire {
namespace event {

metrics_aggregator::metrics_aggregator()
{
  // Initialize histogram buckets to zero
  for (auto& bucket : m_size_histogram) {
    bucket.store(0, std::memory_order_relaxed);
  }
}

void metrics_aggregator::record_allocate(void* ptr, std::size_t size, void* allocator_ref)
{
  std::string allocator_name = allocator_ref_to_string(allocator_ref);

  // Update global counters
  m_allocations_total.fetch_add(1, std::memory_order_relaxed);
  m_bytes_allocated.fetch_add(static_cast<std::int64_t>(size), std::memory_order_relaxed);

  // Update histogram
  std::size_t bucket = get_histogram_bucket(size);
  m_size_histogram[bucket].fetch_add(1, std::memory_order_relaxed);
  m_size_sum.fetch_add(size, std::memory_order_relaxed);

  // Update per-allocator metrics and allocation map (mutex-protected)
  {
    std::lock_guard<std::mutex> lock(m_allocator_mutex);
    auto& metrics = m_per_allocator[allocator_name];
    metrics.allocations++;
    metrics.bytes += static_cast<std::int64_t>(size);

    // Track allocation for later lookup during deallocation
    m_allocation_map[ptr] = {size, allocator_name};
  }
}

void metrics_aggregator::record_deallocate(void* ptr, void* allocator_ref)
{
  std::string allocator_name = allocator_ref_to_string(allocator_ref);
  std::size_t size = 0;

  // Update global counters and per-allocator metrics (mutex-protected)
  {
    std::lock_guard<std::mutex> lock(m_allocator_mutex);

    // Look up allocation size from map
    auto it = m_allocation_map.find(ptr);
    if (it != m_allocation_map.end()) {
      size = it->second.first;
      allocator_name = it->second.second;  // Use allocator from allocation (more accurate)
      m_allocation_map.erase(it);
    }

    m_deallocations_total.fetch_add(1, std::memory_order_relaxed);
    if (size > 0) {
      m_bytes_allocated.fetch_sub(static_cast<std::int64_t>(size), std::memory_order_relaxed);
    }

    auto& metrics = m_per_allocator[allocator_name];
    metrics.deallocations++;
    if (size > 0) {
      metrics.bytes -= static_cast<std::int64_t>(size);
    }
  }
}

std::size_t metrics_aggregator::get_histogram_bucket(std::size_t size)
{
  // Buckets (exponential): <1KB, <4KB, <16KB, <64KB, <256KB, <1MB, <4MB, <16MB, +Inf
  static const std::size_t buckets[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};

  for (std::size_t i = 0; i < 8; ++i) {
    if (size < buckets[i]) {
      return i;
    }
  }
  return 8;  // +Inf bucket
}

std::size_t metrics_aggregator::get_bucket_upper_bound(std::size_t bucket_index)
{
  static const std::size_t buckets[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
  if (bucket_index < 8) {
    return buckets[bucket_index];
  }
  return std::numeric_limits<std::size_t>::max();  // +Inf
}

std::string metrics_aggregator::render_prometheus_text() const
{
  std::ostringstream ss;

  // Timestamp (current time in milliseconds)
  auto now = std::chrono::system_clock::now();
  auto timestamp_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  // Global counters
  ss << "# HELP umpire_allocations_total Total number of allocations\n";
  ss << "# TYPE umpire_allocations_total counter\n";
  ss << "umpire_allocations_total " << m_allocations_total.load(std::memory_order_relaxed) << " "
     << timestamp_ms << "\n";

  ss << "# HELP umpire_deallocations_total Total number of deallocations\n";
  ss << "# TYPE umpire_deallocations_total counter\n";
  ss << "umpire_deallocations_total " << m_deallocations_total.load(std::memory_order_relaxed) << " "
     << timestamp_ms << "\n";

  // Global gauge
  ss << "# HELP umpire_bytes_allocated Current bytes allocated\n";
  ss << "# TYPE umpire_bytes_allocated gauge\n";
  ss << "umpire_bytes_allocated " << m_bytes_allocated.load(std::memory_order_relaxed) << " "
     << timestamp_ms << "\n";

  // Histogram
  ss << "# HELP umpire_allocation_size_bytes Histogram of allocation sizes\n";
  ss << "# TYPE umpire_allocation_size_bytes histogram\n";
  std::uint64_t cumulative = 0;
  for (std::size_t i = 0; i < 9; ++i) {
    cumulative += m_size_histogram[i].load(std::memory_order_relaxed);
    if (i < 8) {
      ss << "umpire_allocation_size_bytes_bucket{le=\"" << get_bucket_upper_bound(i) << "\"} "
         << cumulative << " " << timestamp_ms << "\n";
    } else {
      ss << "umpire_allocation_size_bytes_bucket{le=\"+Inf\"} " << cumulative << " " << timestamp_ms
         << "\n";
    }
  }
  ss << "umpire_allocation_size_bytes_sum " << m_size_sum.load(std::memory_order_relaxed) << " "
     << timestamp_ms << "\n";
  ss << "umpire_allocation_size_bytes_count " << m_allocations_total.load(std::memory_order_relaxed)
     << " " << timestamp_ms << "\n";

  // Per-allocator metrics
  {
    std::lock_guard<std::mutex> lock(m_allocator_mutex);

    if (!m_per_allocator.empty()) {
      ss << "# HELP umpire_allocations_by_allocator Allocations per allocator\n";
      ss << "# TYPE umpire_allocations_by_allocator counter\n";
      for (const auto& [name, metrics] : m_per_allocator) {
        ss << "umpire_allocations_by_allocator{allocator=\"" << name << "\"} " << metrics.allocations
           << " " << timestamp_ms << "\n";
      }

      ss << "# HELP umpire_deallocations_by_allocator Deallocations per allocator\n";
      ss << "# TYPE umpire_deallocations_by_allocator counter\n";
      for (const auto& [name, metrics] : m_per_allocator) {
        ss << "umpire_deallocations_by_allocator{allocator=\"" << name << "\"} "
           << metrics.deallocations << " " << timestamp_ms << "\n";
      }

      ss << "# HELP umpire_bytes_by_allocator Current bytes per allocator\n";
      ss << "# TYPE umpire_bytes_by_allocator gauge\n";
      for (const auto& [name, metrics] : m_per_allocator) {
        ss << "umpire_bytes_by_allocator{allocator=\"" << name << "\"} " << metrics.bytes << " "
           << timestamp_ms << "\n";
      }
    }
  }

  return ss.str();
}

std::vector<std::uint8_t> metrics_aggregator::render_remote_write(int rank, const std::string& job_name,
                                                                   const std::string& environment) const
{
  // TODO: Implement Prometheus remote-write protobuf format
  // This requires:
  // 1. Prometheus remote.proto definitions
  // 2. Protobuf library integration
  // 3. Snappy compression
  //
  // For now, return empty vector. This will be implemented in the next phase.
  // The prometheus_backend can fall back to sending text format via HTTP POST
  // to /api/v1/write endpoint (some Prometheus-compatible receivers support this).

  (void)rank;
  (void)job_name;
  (void)environment;

  return {};
}

std::string metrics_aggregator::allocator_ref_to_string(void* allocator_ref)
{
  std::ostringstream ss;
  ss << allocator_ref;
  return ss.str();
}

} // namespace event
} // namespace umpire
