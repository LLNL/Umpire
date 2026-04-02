//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/memory.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

class test_memory : public umpire::memory {
public:
  explicit test_memory(const std::string& name)
    : umpire::memory{name}
  {
  }

  void* allocate(std::size_t size) override
  {
    void* ptr = std::malloc(size);
    track_allocation(ptr, size);
    return ptr;
  }

  void deallocate(void* ptr) override
  {
    untrack_allocation(ptr);
    std::free(ptr);
  }

  umpire::resource::Platform get_platform() const override
  {
    return umpire::resource::Platform::host;
  }
};

} // namespace

TEST(MemoryThreading, ConcurrentReadOnlyIntrospectionIsStable)
{
  test_memory mem("threading_memory");

  void* a = mem.allocate(128);
  void* b = mem.allocate(256);

  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);

  const int expected_id = mem.get_id();
  const std::string expected_name = mem.get_name();
  const std::size_t expected_current = 384;
  const std::size_t expected_actual = 384;
  const std::size_t expected_highwatermark = 384;
  const auto expected_platform = umpire::resource::Platform::host;

  constexpr int num_threads = 8;
  constexpr int iterations = 50000;

  std::atomic<bool> mismatch{false};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      for (int iter = 0; iter < iterations; ++iter) {
        if (mem.get_id() != expected_id ||
            mem.get_name() != expected_name ||
            mem.get_current_size() != expected_current ||
            mem.get_actual_size() != expected_actual ||
            mem.get_highwatermark() != expected_highwatermark ||
            mem.get_platform() != expected_platform) {
          mismatch.store(true, std::memory_order_relaxed);
          break;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_FALSE(mismatch.load(std::memory_order_relaxed));

  mem.deallocate(a);
  mem.deallocate(b);
}
