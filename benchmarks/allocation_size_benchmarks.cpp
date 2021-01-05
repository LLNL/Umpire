////////////////////////////////////////////////////////////////////////////
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
static const int Max_Allocations{1000000};

template <const int size>
class AllocationSizeBenchmark : public benchmark::Fixture
{
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  AllocationSizeBenchmark() {}

  void SetUp(benchmark::State&) {
    auto& rm = umpire::ResourceManager::getInstance();
    m_alloc = rm.getAllocator("HOST");
  }

  void TearDown(benchmark::State&) {
  }

  void SameOrder(benchmark::State& st) {
    int i{0};
    while (st.KeepRunning()) {
      m_allocations[i++] = m_alloc.allocate(size);
      if (i == Max_Allocations) {
        for (int j{0}; j < i; j++) {
          m_alloc.deallocate(m_allocations[j]);
        }
        i = 0;  
      }
    }
    for (int j{0}; j < i; j++) {
      m_alloc.deallocate(m_allocations[j]);
    }
  }

  void ReverseOrder(benchmark::State& st) {
    int i{0};
    while (st.KeepRunning()) {
      m_allocations[i++] = m_alloc.allocate(size);
      if (i == Max_Allocations) {
        for (int j{i-1}; j >= 0; j--) {
          m_alloc.deallocate(m_allocations[j]);
        }
        i = 0;
      }
    }
    for (int j{i-1};  j >= 0; j--) {
      m_alloc.deallocate(m_allocations[j]);
    }
  }

private: 
  umpire::Allocator m_alloc;
  void* m_allocations[Max_Allocations];
};

class AllocationSize128 : public AllocationSizeBenchmark<128> {};
BENCHMARK_DEFINE_F(AllocationSize128, same_order128)(benchmark::State& st) { SameOrder(st); }
BENCHMARK_DEFINE_F(AllocationSize128, reverse_order128)(benchmark::State& st) { ReverseOrder(st); }

class AllocationSize512 : public AllocationSizeBenchmark<512> {};
BENCHMARK_DEFINE_F(AllocationSize512, same_order512)(benchmark::State& st) { SameOrder(st); }
BENCHMARK_DEFINE_F(AllocationSize512, reverse_order512)(benchmark::State& st) { ReverseOrder(st); }

class AllocationSize1024 : public AllocationSizeBenchmark<1024> {};
BENCHMARK_DEFINE_F(AllocationSize1024, same_order1024)(benchmark::State& st) { SameOrder(st); }
BENCHMARK_DEFINE_F(AllocationSize1024, reverse_order1024)(benchmark::State& st) { ReverseOrder(st); }

class AllocationSize2048 : public AllocationSizeBenchmark<2048> {};
BENCHMARK_DEFINE_F(AllocationSize2048, same_order2048)(benchmark::State& st) { SameOrder(st); }
BENCHMARK_DEFINE_F(AllocationSize2048, reverse_order2048)(benchmark::State& st) { ReverseOrder(st); }

BENCHMARK_REGISTER_F(AllocationSize128, same_order128)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(AllocationSize128, reverse_order128)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(AllocationSize512, same_order512)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(AllocationSize512, reverse_order512)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(AllocationSize1024, same_order1024)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(AllocationSize1024, reverse_order1024)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(AllocationSize2048, same_order2048)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(AllocationSize2048, reverse_order2048)->Range(RangeLow, RangeHi);

BENCHMARK_MAIN();
