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

class NoOpAllocatorBenchmark : public benchmark::Fixture
{
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  NoOpAllocatorBenchmark() {}

  void SetUp(benchmark::State&) {
    auto& rm = umpire::ResourceManager::getInstance();
    alloc = rm.getAllocator("NO_OP");
  }
 
  void TearDown(benchmark::State&) {
  }
  
  void NoOpAllocDealloc(benchmark::State& st) {
    size = static_cast<std::size_t>(st.range(0));
    while (st.KeepRunning()) {
      allocation = alloc.allocate(size);
      alloc.deallocate(allocation);
    }
  }
 
private: 
  std::size_t size;
  umpire::Allocator alloc;
  void* allocation;
};

class NoOpResource : public NoOpAllocatorBenchmark {};
BENCHMARK_DEFINE_F(NoOpResource, allocate_deallocate)(benchmark::State& st) { NoOpAllocDealloc(st); }
BENCHMARK_REGISTER_F(NoOpResource, allocate_deallocate)->Range(RangeLow, RangeHi);

BENCHMARK_MAIN();
