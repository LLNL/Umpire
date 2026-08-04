//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

// Measures allocate/deallocate round-trip latency for v1 allocators that
// MAY be delegating to API v2 under the hood (see UMPIRE_V1_DELEGATE_TO_V2 /
// src/umpire/strategy/detail/v1_backed_memory.hpp) versus their native v1
// implementations, and versus a direct v2 host_memory<> baseline.
//
// This binary builds identically in both flag configurations; it always
// measures whatever ThreadSafeAllocator/SizeLimiter the build provides
// (delegated-to-v2 or native-v1). The flag comparison happens ACROSS builds:
// run this binary once from a build configured with UMPIRE_V1_DELEGATE_TO_V2=Off
// and once with it =On, then diff the two result sets. The header printed at
// startup records which configuration produced the numbers.

#include "umpire/ResourceManager.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

#include <benchmark/benchmark.h>
#include <atomic>
#include <cstdio>
#include <string>

using host_memory = umpire::resource::host_memory<>;

namespace {

std::string unique_allocator_name(const char* prefix)
{
  static std::atomic<int> counter{0};
  return std::string{prefix} + "_" + std::to_string(counter.fetch_add(1));
}

} // namespace

// (a) v1 rm.getAllocator("HOST") raw
static void BM_V1_Host_Raw(benchmark::State& state)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");
  const std::size_t size = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    void* ptr = allocator.allocate(size);
    benchmark::DoNotOptimize(ptr);
    allocator.deallocate(ptr);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_V1_Host_Raw)->Arg(64)->Arg(4096)->Arg(1 << 20);

// (b) v1 ThreadSafeAllocator via makeAllocator (delegated when
// UMPIRE_V1_DELEGATE_TO_V2 is on, native v1 otherwise)
static void BM_V1_ThreadSafeAllocator(benchmark::State& state)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      unique_allocator_name("bench_delegation_thread_safe"), rm.getAllocator("HOST"));
  const std::size_t size = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    void* ptr = allocator.allocate(size);
    benchmark::DoNotOptimize(ptr);
    allocator.deallocate(ptr);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_V1_ThreadSafeAllocator)->Arg(64)->Arg(4096)->Arg(1 << 20);

// (c) v1 SizeLimiter via makeAllocator (delegated when
// UMPIRE_V1_DELEGATE_TO_V2 is on, native v1 otherwise)
static void BM_V1_SizeLimiter(benchmark::State& state)
{
  auto& rm = umpire::ResourceManager::getInstance();
  // Limit set generously above the largest size under test (1MB) so the
  // limiter never actually rejects an allocation.
  auto allocator = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      unique_allocator_name("bench_delegation_size_limiter"), rm.getAllocator("HOST"),
      std::size_t{64} * 1024 * 1024);
  const std::size_t size = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    void* ptr = allocator.allocate(size);
    benchmark::DoNotOptimize(ptr);
    allocator.deallocate(ptr);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_V1_SizeLimiter)->Arg(64)->Arg(4096)->Arg(1 << 20);

// (d) v2 host_memory<> direct
static void BM_V2_HostMemory_Direct(benchmark::State& state)
{
  host_memory mem("bench_delegation_v2_host");
  const std::size_t size = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    void* ptr = mem.allocate(size);
    benchmark::DoNotOptimize(ptr);
    mem.deallocate(ptr);
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(size));
}
BENCHMARK(BM_V2_HostMemory_Direct)->Arg(64)->Arg(4096)->Arg(1 << 20);

int main(int argc, char** argv)
{
#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  std::puts("==============================================================");
  std::puts("api_v2_delegation_benchmarks: UMPIRE_V1_DELEGATE_TO_V2 = ON");
  std::puts("  (v1 ThreadSafeAllocator/SizeLimiter forward to v2 delegates)");
  std::puts("==============================================================");
#else
  std::puts("==============================================================");
  std::puts("api_v2_delegation_benchmarks: UMPIRE_V1_DELEGATE_TO_V2 = OFF");
  std::puts("  (v1 ThreadSafeAllocator/SizeLimiter use native v1 implementations)");
  std::puts("==============================================================");
#endif

  // Touch the v1 HOST allocator once up front so its lazy construction cost
  // isn't attributed to the first benchmark case.
  static_cast<void>(umpire::ResourceManager::getInstance().getAllocator("HOST"));

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
