//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/ResourceManager.hpp"
#include "umpire/TypedAllocator.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/strategy/fixed_pool.hpp"
#include "umpire/strategy/named.hpp"
#include "umpire/strategy/thread_safe.hpp"
#include "../include/umpire/allocator.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <type_traits>
#include <vector>

namespace {

using tracked_host_memory = umpire::resource::host_memory<>;
using fast_host_memory = umpire::resource::fast_host_memory;

template <typename Memory>
void increment_with_platform_dispatch(int* data, std::size_t count)
{
  if constexpr (std::is_same_v<typename Memory::platform, umpire::host_platform>) {
    for (std::size_t i = 0; i < count; ++i) {
      data[i] += 1;
    }
  }
}

void increment_direct(int* data, std::size_t count)
{
  for (std::size_t i = 0; i < count; ++i) {
    data[i] += 1;
  }
}

using pair_type = std::pair<const int, double>;

} // namespace

static void BM_VectorResize_StdAllocator(benchmark::State& state)
{
  const std::size_t count = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    std::vector<int> values;
    values.resize(count);
    benchmark::DoNotOptimize(values.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
}
BENCHMARK(BM_VectorResize_StdAllocator)->Range(64, 1 << 16);

static void BM_VectorResize_V1TypedAllocator(benchmark::State& state)
{
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  auto host = umpire::ResourceManager::getInstance().getAllocator("HOST");
  umpire::TypedAllocator<int> alloc{host};

  for (auto _ : state) {
    std::vector<int, umpire::TypedAllocator<int>> values{alloc};
    values.resize(count);
    benchmark::DoNotOptimize(values.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
}
BENCHMARK(BM_VectorResize_V1TypedAllocator)->Range(64, 1 << 16);

static void BM_VectorResize_V2TrackedAllocator(benchmark::State& state)
{
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  tracked_host_memory host{"TRACKED_VECTOR_HOST"};
  umpire::allocator<int, tracked_host_memory> alloc{&host};

  for (auto _ : state) {
    std::vector<int, umpire::allocator<int, tracked_host_memory>> values{alloc};
    values.resize(count);
    benchmark::DoNotOptimize(values.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
}
BENCHMARK(BM_VectorResize_V2TrackedAllocator)->Range(64, 1 << 16);

static void BM_VectorResize_V2UntrackedAllocator(benchmark::State& state)
{
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  fast_host_memory host{"FAST_VECTOR_HOST"};
  umpire::allocator<int, fast_host_memory> alloc{&host};

  for (auto _ : state) {
    std::vector<int, umpire::allocator<int, fast_host_memory>> values{alloc};
    values.resize(count);
    benchmark::DoNotOptimize(values.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
}
BENCHMARK(BM_VectorResize_V2UntrackedAllocator)->Range(64, 1 << 16);

static void BM_MapInsert_StdAllocator(benchmark::State& state)
{
  const int count = static_cast<int>(state.range(0));

  for (auto _ : state) {
    std::map<int, double> values;
    for (int i = 0; i < count; ++i) {
      values.emplace(i, static_cast<double>(i) * 0.5);
    }
    benchmark::DoNotOptimize(values.size());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_MapInsert_StdAllocator)->Range(64, 4096);

static void BM_MapInsert_V2TrackedAllocator(benchmark::State& state)
{
  const int count = static_cast<int>(state.range(0));
  tracked_host_memory host{"TRACKED_MAP_HOST"};
  umpire::allocator<pair_type, tracked_host_memory> alloc{&host};

  for (auto _ : state) {
    std::map<int, double, std::less<int>, umpire::allocator<pair_type, tracked_host_memory>> values{
      std::less<int>{}, alloc};
    for (int i = 0; i < count; ++i) {
      values.emplace(i, static_cast<double>(i) * 0.5);
    }
    benchmark::DoNotOptimize(values.size());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_MapInsert_V2TrackedAllocator)->Range(64, 4096);

static void BM_MapInsert_V2UntrackedAllocator(benchmark::State& state)
{
  const int count = static_cast<int>(state.range(0));
  fast_host_memory host{"FAST_MAP_HOST"};
  umpire::allocator<pair_type, fast_host_memory> alloc{&host};

  for (auto _ : state) {
    std::map<int, double, std::less<int>, umpire::allocator<pair_type, fast_host_memory>> values{
      std::less<int>{}, alloc};
    for (int i = 0; i < count; ++i) {
      values.emplace(i, static_cast<double>(i) * 0.5);
    }
    benchmark::DoNotOptimize(values.size());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_MapInsert_V2UntrackedAllocator)->Range(64, 4096);

static void BM_Composition_UntrackedHost(benchmark::State& state)
{
  fast_host_memory host{"FAST_HOST"};
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    void* ptr = host.allocate(bytes);
    benchmark::DoNotOptimize(ptr);
    host.deallocate(ptr);
  }

  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bytes));
}
BENCHMARK(BM_Composition_UntrackedHost)->Range(32, 4096);

static void BM_Composition_NamedHost(benchmark::State& state)
{
  fast_host_memory host{"FAST_HOST"};
  umpire::strategy::named<fast_host_memory> named{"NAMED_FAST_HOST", &host};
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    void* ptr = named.allocate(bytes);
    benchmark::DoNotOptimize(ptr);
    named.deallocate(ptr);
  }

  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bytes));
}
BENCHMARK(BM_Composition_NamedHost)->Range(32, 4096);

static void BM_Composition_ThreadSafeNamedHost(benchmark::State& state)
{
  fast_host_memory host{"FAST_HOST"};
  umpire::strategy::named<fast_host_memory> named{"NAMED_FAST_HOST", &host};
  umpire::strategy::thread_safe<umpire::strategy::named<fast_host_memory>> safe{"SAFE_NAMED_FAST_HOST", &named};
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    void* ptr = safe.allocate(bytes);
    benchmark::DoNotOptimize(ptr);
    safe.deallocate(ptr);
  }

  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bytes));
}
BENCHMARK(BM_Composition_ThreadSafeNamedHost)->Range(32, 4096);

static void BM_Composition_ThreadSafeFixedPool(benchmark::State& state)
{
  fast_host_memory host{"FAST_POOL_HOST"};
  umpire::strategy::fixed_pool<fast_host_memory> pool{"FAST_FIXED_POOL", &host, 64, 512};
  umpire::strategy::thread_safe<umpire::strategy::fixed_pool<fast_host_memory>> safe{"SAFE_FAST_FIXED_POOL", &pool};

  for (auto _ : state) {
    void* ptr = safe.allocate(64);
    benchmark::DoNotOptimize(ptr);
    safe.deallocate(ptr);
  }

  state.SetBytesProcessed(state.iterations() * std::int64_t{64});
}
BENCHMARK(BM_Composition_ThreadSafeFixedPool);

static void BM_Dispatch_DirectHostLoop(benchmark::State& state)
{
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  std::vector<int> values(count, 0);

  for (auto _ : state) {
    std::fill(values.begin(), values.end(), 0);
    increment_direct(values.data(), values.size());
    benchmark::DoNotOptimize(values.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
}
BENCHMARK(BM_Dispatch_DirectHostLoop)->Range(256, 1 << 16);

static void BM_Dispatch_CompileTimeHostLoop(benchmark::State& state)
{
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  std::vector<int> values(count, 0);

  for (auto _ : state) {
    std::fill(values.begin(), values.end(), 0);
    increment_with_platform_dispatch<tracked_host_memory>(values.data(), values.size());
    benchmark::DoNotOptimize(values.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(count));
}
BENCHMARK(BM_Dispatch_CompileTimeHostLoop)->Range(256, 1 << 16);

BENCHMARK_MAIN();
