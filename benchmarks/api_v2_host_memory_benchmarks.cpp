//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/host_memory.hpp"

#include <benchmark/benchmark.h>
#include <cstdlib>
#include <vector>

using namespace umpire::resource;

// Benchmark: Raw malloc/free baseline
static void BM_RawMalloc(benchmark::State& state)
{
  const std::size_t size = state.range(0);

  for (auto _ : state) {
    void* ptr = std::malloc(size);
    benchmark::DoNotOptimize(ptr);
    std::free(ptr);
  }

  state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_RawMalloc)->Range(64, 1 << 20);

// Benchmark: host_memory with tracking enabled (default)
static void BM_HostMemory_Tracked(benchmark::State& state)
{
  const std::size_t size = state.range(0);
  host_memory<malloc_allocator, true> mem("BENCH_TRACKED");

  for (auto _ : state) {
    void* ptr = mem.allocate(size);
    benchmark::DoNotOptimize(ptr);
    mem.deallocate(ptr);
  }

  state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_HostMemory_Tracked)->Range(64, 1 << 20);

// Benchmark: host_memory with tracking disabled (fast path)
static void BM_HostMemory_Untracked(benchmark::State& state)
{
  const std::size_t size = state.range(0);
  host_memory<malloc_allocator, false> mem("BENCH_UNTRACKED");

  for (auto _ : state) {
    void* ptr = mem.allocate(size);
    benchmark::DoNotOptimize(ptr);
    mem.deallocate(ptr);
  }

  state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_HostMemory_Untracked)->Range(64, 1 << 20);

// Benchmark: Allocation only (tracked)
static void BM_HostMemory_AllocateOnly_Tracked(benchmark::State& state)
{
  const std::size_t size = state.range(0);
  host_memory<malloc_allocator, true> mem("BENCH_ALLOC_TRACKED");
  std::vector<void*> ptrs;
  ptrs.reserve(10000);

  for (auto _ : state) {
    void* ptr = mem.allocate(size);
    benchmark::DoNotOptimize(ptr);
    ptrs.push_back(ptr);

    if (ptrs.size() >= 10000) {
      state.PauseTiming();
      for (void* p : ptrs) {
        mem.deallocate(p);
      }
      ptrs.clear();
      state.ResumeTiming();
    }
  }

  for (void* p : ptrs) {
    mem.deallocate(p);
  }

  state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_HostMemory_AllocateOnly_Tracked)->Range(64, 1 << 20);

// Benchmark: Allocation only (untracked)
static void BM_HostMemory_AllocateOnly_Untracked(benchmark::State& state)
{
  const std::size_t size = state.range(0);
  host_memory<malloc_allocator, false> mem("BENCH_ALLOC_UNTRACKED");
  std::vector<void*> ptrs;
  ptrs.reserve(10000);

  for (auto _ : state) {
    void* ptr = mem.allocate(size);
    benchmark::DoNotOptimize(ptr);
    ptrs.push_back(ptr);

    if (ptrs.size() >= 10000) {
      state.PauseTiming();
      for (void* p : ptrs) {
        mem.deallocate(p);
      }
      ptrs.clear();
      state.ResumeTiming();
    }
  }

  for (void* p : ptrs) {
    mem.deallocate(p);
  }

  state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_HostMemory_AllocateOnly_Untracked)->Range(64, 1 << 20);

// Benchmark: Deallocation only (tracked)
static void BM_HostMemory_DeallocateOnly_Tracked(benchmark::State& state)
{
  const std::size_t size = state.range(0);
  host_memory<malloc_allocator, true> mem("BENCH_DEALLOC_TRACKED");
  std::vector<void*> ptrs;

  for (auto _ : state) {
    state.PauseTiming();
    ptrs.clear();
    for (int i = 0; i < 1000; ++i) {
      ptrs.push_back(mem.allocate(size));
    }
    state.ResumeTiming();

    for (void* p : ptrs) {
      mem.deallocate(p);
      benchmark::ClobberMemory();
    }
  }
}
BENCHMARK(BM_HostMemory_DeallocateOnly_Tracked)->Range(64, 1 << 20);

// Benchmark: Deallocation only (untracked)
static void BM_HostMemory_DeallocateOnly_Untracked(benchmark::State& state)
{
  const std::size_t size = state.range(0);
  host_memory<malloc_allocator, false> mem("BENCH_DEALLOC_UNTRACKED");
  std::vector<void*> ptrs;

  for (auto _ : state) {
    state.PauseTiming();
    ptrs.clear();
    for (int i = 0; i < 1000; ++i) {
      ptrs.push_back(mem.allocate(size));
    }
    state.ResumeTiming();

    for (void* p : ptrs) {
      mem.deallocate(p);
      benchmark::ClobberMemory();
    }
  }
}
BENCHMARK(BM_HostMemory_DeallocateOnly_Untracked)->Range(64, 1 << 20);

// Benchmark: Many small allocations (tracked)
static void BM_HostMemory_ManySmall_Tracked(benchmark::State& state)
{
  host_memory<malloc_allocator, true> mem("BENCH_MANY_TRACKED");
  const std::size_t alloc_size = 64;
  const int num_allocs = state.range(0);

  std::vector<void*> ptrs;
  ptrs.resize(num_allocs);

  for (auto _ : state) {
    for (int i = 0; i < num_allocs; ++i) {
      ptrs[i] = mem.allocate(alloc_size);
    }
    for (int i = 0; i < num_allocs; ++i) {
      mem.deallocate(ptrs[i]);
    }
  }

  state.SetBytesProcessed(state.iterations() * num_allocs * alloc_size);
}
BENCHMARK(BM_HostMemory_ManySmall_Tracked)->Range(10, 10000);

// Benchmark: Many small allocations (untracked)
static void BM_HostMemory_ManySmall_Untracked(benchmark::State& state)
{
  host_memory<malloc_allocator, false> mem("BENCH_MANY_UNTRACKED");
  const std::size_t alloc_size = 64;
  const int num_allocs = state.range(0);

  std::vector<void*> ptrs;
  ptrs.resize(num_allocs);

  for (auto _ : state) {
    for (int i = 0; i < num_allocs; ++i) {
      ptrs[i] = mem.allocate(alloc_size);
    }
    for (int i = 0; i < num_allocs; ++i) {
      mem.deallocate(ptrs[i]);
    }
  }

  state.SetBytesProcessed(state.iterations() * num_allocs * alloc_size);
}
BENCHMARK(BM_HostMemory_ManySmall_Untracked)->Range(10, 10000);

// Benchmark: Few large allocations (tracked)
static void BM_HostMemory_FewLarge_Tracked(benchmark::State& state)
{
  host_memory<malloc_allocator, true> mem("BENCH_LARGE_TRACKED");
  const std::size_t alloc_size = 10 * 1024 * 1024; // 10MB
  const int num_allocs = state.range(0);

  std::vector<void*> ptrs;
  ptrs.resize(num_allocs);

  for (auto _ : state) {
    for (int i = 0; i < num_allocs; ++i) {
      ptrs[i] = mem.allocate(alloc_size);
    }
    for (int i = 0; i < num_allocs; ++i) {
      mem.deallocate(ptrs[i]);
    }
  }

  state.SetBytesProcessed(state.iterations() * num_allocs * alloc_size);
}
BENCHMARK(BM_HostMemory_FewLarge_Tracked)->Range(1, 10);

// Benchmark: Few large allocations (untracked)
static void BM_HostMemory_FewLarge_Untracked(benchmark::State& state)
{
  host_memory<malloc_allocator, false> mem("BENCH_LARGE_UNTRACKED");
  const std::size_t alloc_size = 10 * 1024 * 1024; // 10MB
  const int num_allocs = state.range(0);

  std::vector<void*> ptrs;
  ptrs.resize(num_allocs);

  for (auto _ : state) {
    for (int i = 0; i < num_allocs; ++i) {
      ptrs[i] = mem.allocate(alloc_size);
    }
    for (int i = 0; i < num_allocs; ++i) {
      mem.deallocate(ptrs[i]);
    }
  }

  state.SetBytesProcessed(state.iterations() * num_allocs * alloc_size);
}
BENCHMARK(BM_HostMemory_FewLarge_Untracked)->Range(1, 10);

// Benchmark: Mixed allocation sizes (tracked)
static void BM_HostMemory_Mixed_Tracked(benchmark::State& state)
{
  host_memory<malloc_allocator, true> mem("BENCH_MIXED_TRACKED");

  std::vector<std::size_t> sizes = {64, 256, 1024, 4096, 16384, 65536};
  std::vector<void*> ptrs;
  ptrs.resize(sizes.size());

  for (auto _ : state) {
    for (size_t i = 0; i < sizes.size(); ++i) {
      ptrs[i] = mem.allocate(sizes[i]);
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
      mem.deallocate(ptrs[i]);
    }
  }

  std::size_t total_bytes = 0;
  for (auto size : sizes) total_bytes += size;
  state.SetBytesProcessed(state.iterations() * total_bytes);
}
BENCHMARK(BM_HostMemory_Mixed_Tracked);

// Benchmark: Mixed allocation sizes (untracked)
static void BM_HostMemory_Mixed_Untracked(benchmark::State& state)
{
  host_memory<malloc_allocator, false> mem("BENCH_MIXED_UNTRACKED");

  std::vector<std::size_t> sizes = {64, 256, 1024, 4096, 16384, 65536};
  std::vector<void*> ptrs;
  ptrs.resize(sizes.size());

  for (auto _ : state) {
    for (size_t i = 0; i < sizes.size(); ++i) {
      ptrs[i] = mem.allocate(sizes[i]);
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
      mem.deallocate(ptrs[i]);
    }
  }

  std::size_t total_bytes = 0;
  for (auto size : sizes) total_bytes += size;
  state.SetBytesProcessed(state.iterations() * total_bytes);
}
BENCHMARK(BM_HostMemory_Mixed_Untracked);

// Benchmark: Singleton access overhead
static void BM_HostMemory_SingletonAccess(benchmark::State& state)
{
  for (auto _ : state) {
    auto& mem = host_memory<>::get();
    benchmark::DoNotOptimize(&mem);
  }
}
BENCHMARK(BM_HostMemory_SingletonAccess);

BENCHMARK_MAIN();
