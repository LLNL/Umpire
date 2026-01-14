//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/detail/registry.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <unordered_set>
#include <vector>

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

