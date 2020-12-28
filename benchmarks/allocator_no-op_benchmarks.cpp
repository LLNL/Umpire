//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "benchmark/benchmark.h"
#include "umpire/config.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

static const int RangeLow{4};
static const int RangeHi{1024};

static void NoOpBenchmark(benchmark::State& st) {
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("NO_OP");

  const std::size_t size {static_cast<std::size_t>(st.range(0))};
  void* allocation;
    
  while (st.KeepRunning()) {
    allocation = alloc.allocate(size);
    alloc.deallocate(allocation);
  }
}

BENCHMARK(NoOpBenchmark)->Range(RangeLow, RangeHi);
BENCHMARK(NoOpBenchmark)->Range(RangeLow, RangeHi);
BENCHMARK(NoOpBenchmark)->Range(RangeLow, RangeHi);

BENCHMARK_MAIN();
