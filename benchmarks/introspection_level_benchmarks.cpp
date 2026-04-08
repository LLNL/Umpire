//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <iostream>

#include "benchmark/benchmark.h"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Introspection.hpp"

static const std::size_t NUM_ALLOCATIONS = 10000;
static const std::size_t ALLOC_SIZE = 1024;

// Get current introspection level (set via UMPIRE_INTROSPECTION_LEVEL env var)
static umpire::IntrospectionLevel getCurrentLevel() {
  auto& rm = umpire::ResourceManager::getInstance();
  return rm.getIntrospectionLevel();
}

// Benchmark allocation/deallocation overhead
static void BM_Allocate(benchmark::State& state) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");
  std::vector<void*> ptrs;
  ptrs.reserve(NUM_ALLOCATIONS);

  for (auto _ : state) {
    state.PauseTiming();
    ptrs.clear();
    state.ResumeTiming();

    for (std::size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
      void* ptr = alloc.allocate(ALLOC_SIZE);
      ptrs.push_back(ptr);
    }

    state.PauseTiming();
    for (auto ptr : ptrs) {
      alloc.deallocate(ptr);
    }
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * NUM_ALLOCATIONS);
}
BENCHMARK(BM_Allocate);

// Benchmark hasAllocator queries (Off mode: always returns false, skip)
static void BM_HasAllocator(benchmark::State& state) {
  auto& rm = umpire::ResourceManager::getInstance();

  if (getCurrentLevel() == umpire::IntrospectionLevel::Off) {
    state.SkipWithError("hasAllocator not available in Off mode");
    return;
  }

  auto alloc = rm.getAllocator("HOST");
  std::vector<void*> ptrs;
  ptrs.reserve(NUM_ALLOCATIONS);

  for (std::size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
    ptrs.push_back(alloc.allocate(ALLOC_SIZE));
  }

  for (auto _ : state) {
    for (auto ptr : ptrs) {
      benchmark::DoNotOptimize(rm.hasAllocator(ptr));
    }
  }

  for (auto ptr : ptrs) {
    alloc.deallocate(ptr);
  }

  state.SetItemsProcessed(state.iterations() * NUM_ALLOCATIONS);
}
BENCHMARK(BM_HasAllocator);

// Benchmark getAllocator queries (Off mode: throws, skip)
static void BM_GetAllocator(benchmark::State& state) {
  auto& rm = umpire::ResourceManager::getInstance();

  if (getCurrentLevel() == umpire::IntrospectionLevel::Off) {
    state.SkipWithError("getAllocator not available in Off mode");
    return;
  }

  auto alloc = rm.getAllocator("HOST");
  std::vector<void*> ptrs;
  ptrs.reserve(NUM_ALLOCATIONS);

  for (std::size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
    ptrs.push_back(alloc.allocate(ALLOC_SIZE));
  }

  for (auto _ : state) {
    for (auto ptr : ptrs) {
      benchmark::DoNotOptimize(rm.getAllocator(ptr));
    }
  }

  for (auto ptr : ptrs) {
    alloc.deallocate(ptr);
  }

  state.SetItemsProcessed(state.iterations() * NUM_ALLOCATIONS);
}
BENCHMARK(BM_GetAllocator);

// Benchmark getSize queries (only available in On mode)
static void BM_GetSize(benchmark::State& state) {
  auto& rm = umpire::ResourceManager::getInstance();

  if (getCurrentLevel() != umpire::IntrospectionLevel::On) {
    state.SkipWithError("getSize only available in On mode");
    return;
  }

  auto alloc = rm.getAllocator("HOST");
  std::vector<void*> ptrs;
  ptrs.reserve(NUM_ALLOCATIONS);

  for (std::size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
    ptrs.push_back(alloc.allocate(ALLOC_SIZE));
  }

  for (auto _ : state) {
    for (auto ptr : ptrs) {
      benchmark::DoNotOptimize(rm.getSize(ptr));
    }
  }

  for (auto ptr : ptrs) {
    alloc.deallocate(ptr);
  }

  state.SetItemsProcessed(state.iterations() * NUM_ALLOCATIONS);
}
BENCHMARK(BM_GetSize);

// Benchmark copy operations (Off mode: not available, skip)
static void BM_Copy(benchmark::State& state) {
  auto& rm = umpire::ResourceManager::getInstance();

  if (getCurrentLevel() == umpire::IntrospectionLevel::Off) {
    state.SkipWithError("copy not available in Off mode");
    return;
  }

  auto alloc = rm.getAllocator("HOST");
  void* src = alloc.allocate(ALLOC_SIZE);
  void* dst = alloc.allocate(ALLOC_SIZE);

  for (auto _ : state) {
    rm.copy(dst, src, ALLOC_SIZE);
  }

  alloc.deallocate(src);
  alloc.deallocate(dst);

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * ALLOC_SIZE);
}
BENCHMARK(BM_Copy);

// Benchmark memset operations (Off mode: not available, skip)
static void BM_Memset(benchmark::State& state) {
  auto& rm = umpire::ResourceManager::getInstance();

  if (getCurrentLevel() == umpire::IntrospectionLevel::Off) {
    state.SkipWithError("memset not available in Off mode");
    return;
  }

  auto alloc = rm.getAllocator("HOST");
  void* ptr = alloc.allocate(ALLOC_SIZE);

  for (auto _ : state) {
    rm.memset(ptr, 0, ALLOC_SIZE);
  }

  alloc.deallocate(ptr);

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * ALLOC_SIZE);
}
BENCHMARK(BM_Memset);

// Benchmark memory overhead (allocation records storage)
// Tests how performance scales with increasing allocation count
static void BM_MemoryOverhead(benchmark::State& state) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");
  std::vector<void*> ptrs;

  for (auto _ : state) {
    state.PauseTiming();
    ptrs.clear();
    ptrs.reserve(state.range(0));
    state.ResumeTiming();

    for (int64_t i = 0; i < state.range(0); ++i) {
      ptrs.push_back(alloc.allocate(ALLOC_SIZE));
    }

    state.PauseTiming();
    for (auto ptr : ptrs) {
      alloc.deallocate(ptr);
    }
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_MemoryOverhead)->Range(100, 100000);

int main(int argc, char** argv) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto level = rm.getIntrospectionLevel();

  std::cout << "============================================\n";
  std::cout << "Introspection Level Benchmarks\n";
  std::cout << "============================================\n";
  std::cout << "Current level: ";
  switch (level) {
    case umpire::IntrospectionLevel::Off:
      std::cout << "Off (zero overhead, no introspection)\n";
      break;
    case umpire::IntrospectionLevel::Basic:
      std::cout << "Basic (runtime API inference, zero storage)\n";
      break;
    case umpire::IntrospectionLevel::On:
      std::cout << "On (full tracking with metadata)\n";
      break;
  }
  std::cout << "============================================\n\n";

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
