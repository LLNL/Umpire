//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/detail/registry.hpp"
#include "umpire/memory.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

class registered_memory : public umpire::memory {
public:
  explicit registered_memory(const std::string& name)
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

TEST(RegistryThreading, ConcurrentIdGenerationNoDuplicates)
{
  auto& r = umpire::detail::registry::get();

  constexpr int threads = 8;
  constexpr int ids_per_thread = 2000;

  std::vector<int> ids;
  ids.resize(threads * ids_per_thread);

  std::atomic<int> index{0};
  std::vector<std::thread> workers;
  workers.reserve(threads);

  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&]() {
      for (int i = 0; i < ids_per_thread; ++i) {
        ids[index.fetch_add(1, std::memory_order_relaxed)] = r.get_id();
      }
    });
  }

  for (auto& th : workers) {
    th.join();
  }

  std::unordered_set<int> unique(ids.begin(), ids.end());
  EXPECT_EQ(unique.size(), ids.size());
}

TEST(RegistryThreading, ConcurrentAllocatorRegistrationAndLookup)
{
  auto& registry = umpire::detail::registry::get();

  constexpr int threads = 8;
  constexpr int allocators_per_thread = 128;

  std::vector<std::vector<std::unique_ptr<registered_memory>>> keep_alive(threads);
  std::vector<std::thread> workers;
  workers.reserve(threads);

  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t]() {
      auto& allocators = keep_alive[t];
      allocators.reserve(allocators_per_thread);

      for (int i = 0; i < allocators_per_thread; ++i) {
        allocators.emplace_back(std::make_unique<registered_memory>(
          "threading_allocator_" + std::to_string(t) + "_" + std::to_string(i)));
      }
    });
  }

  for (auto& thread : workers) {
    thread.join();
  }

  std::vector<int> ids;
  std::vector<std::string> names;
  ids.reserve(threads * allocators_per_thread);
  names.reserve(threads * allocators_per_thread);

  for (const auto& allocators : keep_alive) {
    for (const auto& allocator : allocators) {
      ids.push_back(allocator->get_id());
      names.push_back(allocator->get_name());

      EXPECT_EQ(registry.find_allocator_by_id(allocator->get_id()), allocator.get());
      EXPECT_EQ(registry.find_allocator_by_name(allocator->get_name()), allocator.get());
    }
  }

  std::unordered_set<int> unique_ids(ids.begin(), ids.end());
  EXPECT_EQ(unique_ids.size(), ids.size());

  keep_alive.clear();

  for (std::size_t i = 0; i < ids.size(); ++i) {
    EXPECT_EQ(registry.find_allocator_by_id(ids[i]), nullptr);
    EXPECT_EQ(registry.find_allocator_by_name(names[i]), nullptr);
  }
}
