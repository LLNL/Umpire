//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/thread_safe.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/memory.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

// Test memory implementation for unit testing
class test_memory : public umpire::memory {
public:
  test_memory() : umpire::memory{"test_parent"} { }

  void* allocate(std::size_t size) override
  {
    void* ptr{std::malloc(size)};
    track_allocation(ptr, size);
    return ptr;
  }

  void deallocate(void* ptr) override
  {
    untrack_allocation(ptr);
    std::free(ptr);
  }

  umpire::resource::Platform get_platform() const override {
    return umpire::resource::Platform::host;
  }
};

// Memory implementation that throws on allocate (for exception testing)
class throwing_memory : public umpire::memory {
public:
  throwing_memory() : umpire::memory{"throwing_parent"} { }

  void* allocate(std::size_t /* size */) override
  {
    throw std::runtime_error("Intentional allocation failure");
  }

  void deallocate(void* /* ptr */) override
  {
    throw std::runtime_error("Intentional deallocation failure");
  }

  umpire::resource::Platform get_platform() const override {
    return umpire::resource::Platform::host;
  }
};

// Memory implementation with instrumentation for testing thread safety
class instrumented_memory : public umpire::memory {
private:
  std::atomic<int> active_operations_{0};
  std::atomic<int> max_concurrent_{0};
  std::atomic<int> allocation_count_{0};
  std::atomic<int> deallocation_count_{0};

public:
  instrumented_memory() : umpire::memory{"instrumented_parent"} { }

  void* allocate(std::size_t size) override
  {
    // Track concurrent operations
    int current = active_operations_.fetch_add(1, std::memory_order_relaxed) + 1;

    // Update max concurrent (may race, but that's ok for testing)
    int max_val = max_concurrent_.load(std::memory_order_relaxed);
    while (current > max_val &&
           !max_concurrent_.compare_exchange_weak(max_val, current, std::memory_order_relaxed)) {
      // Retry if CAS failed
    }

    allocation_count_.fetch_add(1, std::memory_order_relaxed);

    void* ptr{std::malloc(size)};
    track_allocation(ptr, size);

    active_operations_.fetch_sub(1, std::memory_order_relaxed);
    return ptr;
  }

  void deallocate(void* ptr) override
  {
    // Track concurrent operations
    int current = active_operations_.fetch_add(1, std::memory_order_relaxed) + 1;

    // Update max concurrent
    int max_val = max_concurrent_.load(std::memory_order_relaxed);
    while (current > max_val &&
           !max_concurrent_.compare_exchange_weak(max_val, current, std::memory_order_relaxed)) {
      // Retry if CAS failed
    }

    deallocation_count_.fetch_add(1, std::memory_order_relaxed);

    untrack_allocation(ptr);
    std::free(ptr);

    active_operations_.fetch_sub(1, std::memory_order_relaxed);
  }

  umpire::resource::Platform get_platform() const override {
    return umpire::resource::Platform::host;
  }

  int get_max_concurrent() const { return max_concurrent_.load(); }
  int get_allocation_count() const { return allocation_count_.load(); }
  int get_deallocation_count() const { return deallocation_count_.load(); }
  void reset_counters() {
    active_operations_ = 0;
    max_concurrent_ = 0;
    allocation_count_ = 0;
    deallocation_count_ = 0;
  }
};

} // namespace

// ============================================================================
// Construction and Validation Tests
// ============================================================================

TEST(thread_safe, construct_with_valid_parent)
{
  test_memory parent;
  umpire::strategy::thread_safe<test_memory> strategy("thread_safe", &parent);

  EXPECT_EQ(strategy.get_parent(), &parent);
  EXPECT_EQ(strategy.get_name(), "thread_safe");
}

TEST(thread_safe, construct_with_nullptr_throws)
{
  EXPECT_THROW(
    umpire::strategy::thread_safe<test_memory> strategy("thread_safe", nullptr),
    std::invalid_argument
  );
}

TEST(thread_safe, get_platform_delegates_to_parent)
{
  test_memory parent;
  umpire::strategy::thread_safe<test_memory> strategy("thread_safe", &parent);

  EXPECT_EQ(strategy.get_platform(), parent.get_platform());
  EXPECT_EQ(strategy.get_platform(), umpire::resource::Platform::host);
}

// ============================================================================
// Single-Threaded Correctness Tests
// ============================================================================

TEST(thread_safe, single_threaded_basic_allocation)
{
  test_memory parent;
  umpire::strategy::thread_safe<test_memory> strategy("thread_safe", &parent);

  void* ptr = strategy.allocate(64);
  EXPECT_NE(ptr, nullptr);

  // Verify parent's statistics were updated
  EXPECT_EQ(parent.get_current_size(), 64);

  strategy.deallocate(ptr);
  EXPECT_EQ(parent.get_current_size(), 0);
}

TEST(thread_safe, single_threaded_multiple_allocations)
{
  test_memory parent;
  umpire::strategy::thread_safe<test_memory> strategy("thread_safe", &parent);

  void* ptr1 = strategy.allocate(32);
  void* ptr2 = strategy.allocate(64);
  void* ptr3 = strategy.allocate(128);

  EXPECT_EQ(parent.get_current_size(), 224);

  strategy.deallocate(ptr2);
  EXPECT_EQ(parent.get_current_size(), 160);

  strategy.deallocate(ptr1);
  EXPECT_EQ(parent.get_current_size(), 128);

  strategy.deallocate(ptr3);
  EXPECT_EQ(parent.get_current_size(), 0);
}

TEST(thread_safe, single_threaded_zero_size_allocation)
{
  test_memory parent;
  umpire::strategy::thread_safe<test_memory> strategy("thread_safe", &parent);

  void* ptr = strategy.allocate(0);
  // Behavior depends on parent implementation
  strategy.deallocate(ptr);
}

TEST(thread_safe, single_threaded_nullptr_deallocation)
{
  test_memory parent;
  umpire::strategy::thread_safe<test_memory> strategy("thread_safe", &parent);

  // Should not crash
  EXPECT_NO_THROW(strategy.deallocate(nullptr));
}

// ============================================================================
// Exception Safety Tests
// ============================================================================

TEST(thread_safe, exception_propagation_on_allocate)
{
  throwing_memory parent;
  umpire::strategy::thread_safe<throwing_memory> strategy("thread_safe", &parent);

  // Exception should propagate through thread_safe wrapper
  EXPECT_THROW(strategy.allocate(64), std::runtime_error);
}

TEST(thread_safe, exception_propagation_on_deallocate)
{
  throwing_memory parent;
  umpire::strategy::thread_safe<throwing_memory> strategy("thread_safe", &parent);

  // Exception should propagate through thread_safe wrapper
  void* dummy = reinterpret_cast<void*>(0x1000);
  EXPECT_THROW(strategy.deallocate(dummy), std::runtime_error);
}

TEST(thread_safe, mutex_released_after_exception)
{
  throwing_memory parent;
  umpire::strategy::thread_safe<throwing_memory> strategy("thread_safe", &parent);

  // First call throws
  EXPECT_THROW(strategy.allocate(64), std::runtime_error);

  // Second call should also throw (not deadlock)
  // If mutex wasn't released, this would hang
  EXPECT_THROW(strategy.allocate(64), std::runtime_error);
}

// ============================================================================
// Multi-Threaded Stress Tests
// ============================================================================

TEST(thread_safe, multithreaded_concurrent_allocations)
{
  test_memory parent;
  umpire::strategy::thread_safe<test_memory> strategy("thread_safe", &parent);

  constexpr int num_threads = 10;
  constexpr int allocs_per_thread = 1000;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Launch threads that allocate and deallocate concurrently
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy]() {
      std::vector<void*> ptrs;
      ptrs.reserve(allocs_per_thread);

      // Allocate
      for (int i = 0; i < allocs_per_thread; ++i) {
        void* ptr = strategy.allocate(64);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
      }

      // Deallocate
      for (void* ptr : ptrs) {
        strategy.deallocate(ptr);
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // All memory should be returned
  EXPECT_EQ(parent.get_current_size(), 0);
}

TEST(thread_safe, multithreaded_interleaved_operations)
{
  test_memory parent;
  umpire::strategy::thread_safe<test_memory> strategy("thread_safe", &parent);

  constexpr int num_threads = 10;
  constexpr int operations_per_thread = 500;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Launch threads that interleave allocations and deallocations
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy]() {
      std::vector<void*> ptrs;

      for (int i = 0; i < operations_per_thread; ++i) {
        // Allocate
        void* ptr = strategy.allocate(32 + (i % 128));
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);

        // Periodically deallocate some
        if (i > 0 && i % 10 == 0 && !ptrs.empty()) {
          strategy.deallocate(ptrs.back());
          ptrs.pop_back();
        }
      }

      // Cleanup remaining
      for (void* ptr : ptrs) {
        strategy.deallocate(ptr);
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // All memory should be returned
  EXPECT_EQ(parent.get_current_size(), 0);
}

TEST(thread_safe, multithreaded_serialization_verification)
{
  instrumented_memory parent;
  umpire::strategy::thread_safe<instrumented_memory> strategy("thread_safe", &parent);

  constexpr int num_threads = 10;
  constexpr int allocs_per_thread = 100;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Launch threads
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy]() {
      std::vector<void*> ptrs;
      ptrs.reserve(allocs_per_thread);

      for (int i = 0; i < allocs_per_thread; ++i) {
        void* ptr = strategy.allocate(64);
        ptrs.push_back(ptr);
      }

      for (void* ptr : ptrs) {
        strategy.deallocate(ptr);
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // With proper locking, max_concurrent should be 1
  // (only one thread in allocate/deallocate at a time)
  EXPECT_EQ(parent.get_max_concurrent(), 1);

  // Verify all operations completed
  EXPECT_EQ(parent.get_allocation_count(), num_threads * allocs_per_thread);
  EXPECT_EQ(parent.get_deallocation_count(), num_threads * allocs_per_thread);
}

TEST(thread_safe, multithreaded_variable_sizes)
{
  test_memory parent;
  umpire::strategy::thread_safe<test_memory> strategy("thread_safe", &parent);

  constexpr int num_threads = 8;
  constexpr int allocs_per_thread = 500;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&strategy, t]() {
      std::vector<void*> ptrs;
      ptrs.reserve(allocs_per_thread);

      // Each thread uses different size pattern
      for (int i = 0; i < allocs_per_thread; ++i) {
        std::size_t size = 16 * (1 << (i % 6));  // 16, 32, 64, 128, 256, 512
        void* ptr = strategy.allocate(size);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
      }

      // Deallocate in reverse order
      for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
        strategy.deallocate(*it);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(parent.get_current_size(), 0);
}

// ============================================================================
// Platform Type Propagation Tests
// ============================================================================

TEST(thread_safe, platform_type_propagation)
{
  using host_mem = umpire::resource::host_memory<>;
  using thread_safe_host = umpire::strategy::thread_safe<host_mem>;

  // Platform type should be propagated from host_memory
  static_assert(std::is_same<thread_safe_host::platform, umpire::host_platform>::value,
                "Platform type should be propagated from wrapped memory");
}

// ============================================================================
// Composition Tests
// ============================================================================

TEST(thread_safe, composition_with_host_memory)
{
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::thread_safe<umpire::resource::host_memory<>>
    strategy("thread_safe_host", &host);

  void* ptr = strategy.allocate(1024);
  EXPECT_NE(ptr, nullptr);

  // Write to verify memory is accessible
  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[1023] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[1023], 'Z');

  strategy.deallocate(ptr);
}
