//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/thread_safe.hpp"
#include "umpire/resource/host_memory.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

// ============================================================================
// Heavy Stress Scenarios
// ============================================================================

TEST(thread_safe_stress, hammering_stress)
{
  // Hammer the allocator with many threads doing rapid allocations
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("hammering", &host);

  constexpr int num_threads = 20;
  constexpr int iterations = 5000;
  std::atomic<int> total_allocations{0};

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy, &total_allocations]() {
      for (int i = 0; i < iterations; ++i) {
        // Rapid allocate/deallocate
        void* ptr = strategy.allocate(64);
        total_allocations.fetch_add(1, std::memory_order_relaxed);

        // Minimal work to stress the lock more
        volatile char* p = static_cast<char*>(ptr);
        p[0] = 'X';

        strategy.deallocate(ptr);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(total_allocations.load(), num_threads * iterations);
}

TEST(thread_safe_stress, mixed_allocation_sizes)
{
  // Test with highly variable allocation sizes
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("mixed_sizes", &host);

  constexpr int num_threads = 16;
  constexpr int allocs_per_thread = 2000;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy, t]() {
      // Use thread-specific seed for reproducibility
      std::mt19937 rng(t);
      std::uniform_int_distribution<std::size_t> size_dist(1, 16384);

      std::vector<void*> ptrs;
      ptrs.reserve(allocs_per_thread);

      for (int i = 0; i < allocs_per_thread; ++i) {
        std::size_t size = size_dist(rng);
        void* ptr = strategy.allocate(size);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
      }

      // Deallocate in random order
      std::shuffle(ptrs.begin(), ptrs.end(), rng);
      for (void* ptr : ptrs) {
        strategy.deallocate(ptr);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(thread_safe_stress, variable_lifetime_stress)
{
  // Allocations with highly variable lifetimes
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("variable_lifetime", &host);

  constexpr int num_threads = 12;
  constexpr int total_allocations = 3000;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy, t]() {
      std::mt19937 rng(t * 1000);
      std::uniform_int_distribution<int> lifetime_dist(1, 100);
      std::uniform_int_distribution<std::size_t> size_dist(16, 1024);

      struct Allocation {
        void* ptr;
        int lifetime;
      };

      std::vector<Allocation> active;

      for (int i = 0; i < total_allocations; ++i) {
        // Allocate with random lifetime
        std::size_t size = size_dist(rng);
        void* ptr = strategy.allocate(size);
        int lifetime = lifetime_dist(rng);
        active.push_back({ptr, lifetime});

        // Age all allocations and free expired ones
        for (auto it = active.begin(); it != active.end(); ) {
          it->lifetime--;
          if (it->lifetime <= 0) {
            strategy.deallocate(it->ptr);
            it = active.erase(it);
          } else {
            ++it;
          }
        }
      }

      // Cleanup remaining
      for (const auto& alloc : active) {
        strategy.deallocate(alloc.ptr);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(thread_safe_stress, burst_pattern_stress)
{
  // Threads alternate between bursts of activity and idle periods
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("burst_pattern", &host);

  constexpr int num_threads = 10;
  constexpr int num_bursts = 50;
  constexpr int allocs_per_burst = 200;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy]() {
      for (int burst = 0; burst < num_bursts; ++burst) {
        std::vector<void*> ptrs;
        ptrs.reserve(allocs_per_burst);

        // Burst of allocations
        for (int i = 0; i < allocs_per_burst; ++i) {
          void* ptr = strategy.allocate(128);
          ptrs.push_back(ptr);
        }

        // Burst of deallocations
        for (void* ptr : ptrs) {
          strategy.deallocate(ptr);
        }

        // Small idle period (simulate work)
        std::this_thread::yield();
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(thread_safe_stress, long_lived_allocations)
{
  // Mix of short-lived and long-lived allocations
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("long_lived", &host);

  constexpr int num_threads = 8;
  constexpr int short_lived_count = 1000;
  constexpr int long_lived_count = 100;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Shared long-lived allocations
  std::vector<void*> long_lived;
  for (int i = 0; i < long_lived_count; ++i) {
    long_lived.push_back(strategy.allocate(4096));
  }

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy]() {
      // Many short-lived allocations
      for (int i = 0; i < short_lived_count; ++i) {
        void* ptr = strategy.allocate(64);
        // Simulate some work
        volatile char* p = static_cast<char*>(ptr);
        p[0] = 'X';
        strategy.deallocate(ptr);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Cleanup long-lived
  for (void* ptr : long_lived) {
    strategy.deallocate(ptr);
  }
}

// ============================================================================
// Composition Tests
// ============================================================================

namespace {

// Simple passthrough strategy for composition testing
class passthrough_strategy : public umpire::allocation_strategy {
public:
  using platform = umpire::host_platform;

  passthrough_strategy(const std::string& name, umpire::memory* parent)
    : allocation_strategy(name, parent) {}

  void* allocate(std::size_t size) override {
    return parent_->allocate(size);
  }

  void deallocate(void* ptr) override {
    parent_->deallocate(ptr);
  }
};

} // namespace

TEST(thread_safe_stress, composition_thread_safe_wrapping_strategy)
{
  // thread_safe wrapping another strategy
  auto& host = umpire::resource::host_memory<>::get();
  passthrough_strategy inner("passthrough", &host);
  umpire::strategy::thread_safe<passthrough_strategy>
    outer("thread_safe_wrapper", &inner);

  constexpr int num_threads = 10;
  constexpr int allocs_per_thread = 1000;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&outer]() {
      std::vector<void*> ptrs;
      ptrs.reserve(allocs_per_thread);

      for (int i = 0; i < allocs_per_thread; ++i) {
        void* ptr = outer.allocate(64);
        ptrs.push_back(ptr);
      }

      for (void* ptr : ptrs) {
        outer.deallocate(ptr);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(thread_safe_stress, composition_nested_strategies)
{
  // Multiple strategies composed together
  auto& host = umpire::resource::host_memory<>::get();
  passthrough_strategy inner1("inner1", &host);
  passthrough_strategy inner2("inner2", &inner1);
  umpire::strategy::thread_safe<passthrough_strategy>
    outer("thread_safe", &inner2);

  constexpr int num_threads = 8;
  constexpr int allocs_per_thread = 500;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&outer]() {
      for (int i = 0; i < allocs_per_thread; ++i) {
        void* ptr = outer.allocate(128);
        EXPECT_NE(ptr, nullptr);
        outer.deallocate(ptr);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

// ============================================================================
// Realistic Workload Simulation
// ============================================================================

TEST(thread_safe_stress, realistic_mixed_workload)
{
  // Simulate a more realistic workload with different thread behaviors
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("realistic", &host);

  std::atomic<bool> keep_running{true};
  std::atomic<std::size_t> total_ops{0};

  auto heavy_allocator = [&]() {
    // Thread that does heavy allocations
    while (keep_running.load(std::memory_order_relaxed)) {
      std::vector<void*> ptrs;
      for (int i = 0; i < 100; ++i) {
        ptrs.push_back(strategy.allocate(1024));
      }
      for (void* ptr : ptrs) {
        strategy.deallocate(ptr);
      }
      total_ops.fetch_add(100, std::memory_order_relaxed);
    }
  };

  auto light_allocator = [&]() {
    // Thread that does light, frequent allocations
    while (keep_running.load(std::memory_order_relaxed)) {
      for (int i = 0; i < 1000; ++i) {
        void* ptr = strategy.allocate(32);
        strategy.deallocate(ptr);
      }
      total_ops.fetch_add(1000, std::memory_order_relaxed);
    }
  };

  auto bursty_allocator = [&]() {
    // Thread that does occasional bursts
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> sleep_dist(1, 10);

    while (keep_running.load(std::memory_order_relaxed)) {
      std::vector<void*> ptrs;
      for (int i = 0; i < 50; ++i) {
        ptrs.push_back(strategy.allocate(256));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_dist(rng)));
      for (void* ptr : ptrs) {
        strategy.deallocate(ptr);
      }
      total_ops.fetch_add(50, std::memory_order_relaxed);
    }
  };

  // Launch different types of threads
  std::vector<std::thread> threads;
  threads.emplace_back(heavy_allocator);
  threads.emplace_back(heavy_allocator);
  threads.emplace_back(light_allocator);
  threads.emplace_back(light_allocator);
  threads.emplace_back(light_allocator);
  threads.emplace_back(bursty_allocator);

  // Run for a short time
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  keep_running.store(false, std::memory_order_relaxed);

  for (auto& thread : threads) {
    thread.join();
  }

  // Just verify we did a lot of operations without crashing
  EXPECT_GT(total_ops.load(), 0);
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

TEST(thread_safe_stress, many_threads_few_operations)
{
  // Many threads, but each does very few operations
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("many_threads", &host);

  constexpr int num_threads = 100;
  constexpr int operations = 10;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy]() {
      for (int i = 0; i < operations; ++i) {
        void* ptr = strategy.allocate(64);
        strategy.deallocate(ptr);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(thread_safe_stress, few_threads_many_operations)
{
  // Few threads, but each does many operations
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("few_threads", &host);

  constexpr int num_threads = 2;
  constexpr int operations = 50000;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy]() {
      for (int i = 0; i < operations; ++i) {
        void* ptr = strategy.allocate(64);
        strategy.deallocate(ptr);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(thread_safe_stress, maximum_contention)
{
  // All threads try to access at exactly the same time
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("max_contention", &host);

  constexpr int num_threads = 16;
  constexpr int operations = 1000;

  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy, &start]() {
      // Wait for signal
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      // All threads start at once
      for (int i = 0; i < operations; ++i) {
        void* ptr = strategy.allocate(64);
        strategy.deallocate(ptr);
      }
    });
  }

  // Release all threads at once
  start.store(true, std::memory_order_release);

  for (auto& thread : threads) {
    thread.join();
  }
}
