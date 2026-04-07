//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include "umpire/event/metrics_aggregator.hpp"

#include <thread>
#include <vector>

using namespace umpire::event;

TEST(MetricsAggregatorTest, InitialState)
{
  metrics_aggregator agg;

  EXPECT_EQ(agg.allocations_total(), 0);
  EXPECT_EQ(agg.deallocations_total(), 0);
  EXPECT_EQ(agg.bytes_allocated(), 0);
}

TEST(MetricsAggregatorTest, RecordAllocate)
{
  metrics_aggregator agg;

  void* ptr1 = reinterpret_cast<void*>(0x1000);
  void* allocator1 = reinterpret_cast<void*>(0x100);

  agg.record_allocate(ptr1, 1024, allocator1);

  EXPECT_EQ(agg.allocations_total(), 1);
  EXPECT_EQ(agg.deallocations_total(), 0);
  EXPECT_EQ(agg.bytes_allocated(), 1024);
}

TEST(MetricsAggregatorTest, RecordDeallocate)
{
  metrics_aggregator agg;

  void* ptr1 = reinterpret_cast<void*>(0x1000);
  void* allocator1 = reinterpret_cast<void*>(0x100);

  // Allocate first
  agg.record_allocate(ptr1, 1024, allocator1);

  // Then deallocate
  agg.record_deallocate(ptr1, allocator1);

  EXPECT_EQ(agg.allocations_total(), 1);
  EXPECT_EQ(agg.deallocations_total(), 1);
  EXPECT_EQ(agg.bytes_allocated(), 0);
}

TEST(MetricsAggregatorTest, MultipleAllocations)
{
  metrics_aggregator agg;

  void* allocator1 = reinterpret_cast<void*>(0x100);

  for (int i = 0; i < 10; i++) {
    void* ptr = reinterpret_cast<void*>(0x1000 + i * 1024);
    agg.record_allocate(ptr, 512, allocator1);
  }

  EXPECT_EQ(agg.allocations_total(), 10);
  EXPECT_EQ(agg.bytes_allocated(), 5120);
}

TEST(MetricsAggregatorTest, DeallocateUnknownPointer)
{
  metrics_aggregator agg;

  void* ptr1 = reinterpret_cast<void*>(0x1000);
  void* allocator1 = reinterpret_cast<void*>(0x100);

  // Deallocate pointer that was never allocated
  agg.record_deallocate(ptr1, allocator1);

  // Should still count the deallocation, but bytes remain 0
  EXPECT_EQ(agg.deallocations_total(), 1);
  EXPECT_EQ(agg.bytes_allocated(), 0);
}

TEST(MetricsAggregatorTest, MultipleAllocators)
{
  metrics_aggregator agg;

  void* allocator1 = reinterpret_cast<void*>(0x100);
  void* allocator2 = reinterpret_cast<void*>(0x200);

  void* ptr1 = reinterpret_cast<void*>(0x1000);
  void* ptr2 = reinterpret_cast<void*>(0x2000);

  agg.record_allocate(ptr1, 1024, allocator1);
  agg.record_allocate(ptr2, 2048, allocator2);

  EXPECT_EQ(agg.allocations_total(), 2);
  EXPECT_EQ(agg.bytes_allocated(), 3072);
}

TEST(MetricsAggregatorTest, PrometheusTextFormat)
{
  metrics_aggregator agg;

  void* ptr1 = reinterpret_cast<void*>(0x1000);
  void* allocator1 = reinterpret_cast<void*>(0x100);

  agg.record_allocate(ptr1, 1024, allocator1);

  std::string text = agg.render_prometheus_text();

  // Check that essential metrics are present
  EXPECT_NE(text.find("umpire_allocations_total"), std::string::npos);
  EXPECT_NE(text.find("umpire_bytes_allocated"), std::string::npos);
  EXPECT_NE(text.find("# HELP"), std::string::npos);
  EXPECT_NE(text.find("# TYPE"), std::string::npos);
}

TEST(MetricsAggregatorTest, HistogramBuckets)
{
  metrics_aggregator agg;

  void* allocator1 = reinterpret_cast<void*>(0x100);

  // Allocate different sizes to hit different buckets
  std::vector<std::size_t> sizes = {
      512,       // <1KB bucket
      2048,      // <4KB bucket
      8192,      // <16KB bucket
      32768,     // <64KB bucket
      131072,    // <256KB bucket
      524288,    // <1MB bucket
      2097152,   // <4MB bucket
      8388608,   // <16MB bucket
      67108864   // +Inf bucket
  };

  for (std::size_t i = 0; i < sizes.size(); i++) {
    void* ptr = reinterpret_cast<void*>(0x1000 + i * 1024);
    agg.record_allocate(ptr, sizes[i], allocator1);
  }

  std::string text = agg.render_prometheus_text();

  // Check histogram is present
  EXPECT_NE(text.find("umpire_allocation_size_bytes_bucket"), std::string::npos);
  EXPECT_NE(text.find("umpire_allocation_size_bytes_sum"), std::string::npos);
  EXPECT_NE(text.find("umpire_allocation_size_bytes_count"), std::string::npos);
  EXPECT_NE(text.find("le=\"+Inf\""), std::string::npos);
}

TEST(MetricsAggregatorTest, ConcurrentAllocations)
{
  metrics_aggregator agg;

  const int num_threads = 4;
  const int allocs_per_thread = 1000;

  auto allocate_worker = [&](int thread_id) {
    void* allocator = reinterpret_cast<void*>(0x100);
    for (int i = 0; i < allocs_per_thread; i++) {
      void* ptr = reinterpret_cast<void*>((thread_id * 10000 + i) * 1024);
      agg.record_allocate(ptr, 512, allocator);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(allocate_worker, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(agg.allocations_total(), num_threads * allocs_per_thread);
  EXPECT_EQ(agg.bytes_allocated(), num_threads * allocs_per_thread * 512);
}

TEST(MetricsAggregatorTest, ConcurrentAllocateAndDeallocate)
{
  metrics_aggregator agg;

  const int num_threads = 4;
  const int ops_per_thread = 500;

  auto worker = [&](int thread_id) {
    void* allocator = reinterpret_cast<void*>(0x100);
    for (int i = 0; i < ops_per_thread; i++) {
      void* ptr = reinterpret_cast<void*>((thread_id * 10000 + i) * 1024);
      agg.record_allocate(ptr, 1024, allocator);
      agg.record_deallocate(ptr, allocator);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(worker, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(agg.allocations_total(), num_threads * ops_per_thread);
  EXPECT_EQ(agg.deallocations_total(), num_threads * ops_per_thread);
  EXPECT_EQ(agg.bytes_allocated(), 0);  // All deallocated
}

TEST(MetricsAggregatorTest, BytesAllocatedCanGoNegative)
{
  metrics_aggregator agg;

  void* ptr1 = reinterpret_cast<void*>(0x1000);
  void* allocator1 = reinterpret_cast<void*>(0x100);

  // Deallocate without allocating first (shouldn't happen in practice, but test the gauge behavior)
  agg.record_deallocate(ptr1, allocator1);

  // Gauge can go negative (signed int64)
  EXPECT_LE(agg.bytes_allocated(), 0);
}
