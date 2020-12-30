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
static const int Max_Allocations{10};

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
  
  void NoOpSameOrder(benchmark::State& st) {
    size = static_cast<std::size_t>(st.range(0));
    int i{0};
    while (st.KeepRunning()) {
      m_allocations[i++] = alloc.allocate(size);
      if (i == Max_Allocations) {
        for (int j{0}; j < i; j++) 
          alloc.deallocate(m_allocations[j]);
        i = 0;
      }
    }
    for (int j{0}; j < i; j++) {
      alloc.deallocate(m_allocations[j]);
    }
  }
  void NoOpReverseOrder(benchmark::State& st) {
    size = static_cast<std::size_t>(st.range(0));
    int i{0};
    while (st.KeepRunning()) {
      m_allocations[i++] = alloc.allocate(size);
      if (i == Max_Allocations) {
        for (int j{i}; j < 0; j--) 
          alloc.deallocate(m_allocations[j]);
        i = 0;
      }
    }
    for (int j{i};  j < 0; j--) {
      alloc.deallocate(m_allocations[j]);
    }
  }
  void NoOpShuffle(benchmark::State& st) {
    size = static_cast<std::size_t>(st.range(0));
    int i{0};
    while (st.KeepRunning()) {
      m_allocations[i++] = alloc.allocate(size);
      if (i == Max_Allocations) {
        for (int j{0}; j < (i/2); j++) {
          alloc.deallocate(m_allocations[j]);
          alloc.deallocate(m_allocations[(i/2)+j]);
        }
        i = 0;
      }
    }
    //pseudo-random :)
    for(int j{0}; j < (i/2); j++) {
      alloc.deallocate(m_allocations[j]);
      alloc.deallocate(m_allocations[(i/2)+j]);
    }
  }
 
private: 
  std::size_t size;
  umpire::Allocator alloc;
  void* m_allocations[Max_Allocations];
};

class NoOpResource : public NoOpAllocatorBenchmark {};
BENCHMARK_DEFINE_F(NoOpResource, same_order)(benchmark::State& st) { NoOpSameOrder(st); }
BENCHMARK_DEFINE_F(NoOpResource, reverse_order)(benchmark::State& st) { NoOpReverseOrder(st); }
BENCHMARK_DEFINE_F(NoOpResource, shuffle)(benchmark::State& st) { NoOpShuffle(st); }

BENCHMARK_REGISTER_F(NoOpResource, same_order)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(NoOpResource, reverse_order)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(NoOpResource, shuffle)->Range(RangeLow, RangeHi);

BENCHMARK_MAIN();
