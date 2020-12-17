//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <functional>
#include <algorithm>
#include <random>

#include "benchmark/benchmark.h"

#include "umpire/config.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/resource/MemoryResourceTypes.hpp"

#include "umpire/alloc/NoOpAllocator.hpp"

static const int RangeLow{4};
static const int RangeHi{1024};

static const bool Introspection{true};

static const std::size_t Max_Allocations{100000};
static const std::size_t Num_Random{1000};

class AllocatorBenchmark : public benchmark::Fixture {
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  virtual void* allocate(std::size_t nbytes) = 0;
  virtual void deallocate(void* ptr) = 0;

  void largeAllocDealloc(benchmark::State& st) {
    const std::size_t size{
      static_cast<std::size_t>(st.range(0)) * 1024 * 1024 * 1024};
    void* allocation;

    while (st.KeepRunning()) {
      allocation = allocate(size);
      deallocate(allocation);
    }
  }

  void allocation(benchmark::State& st) {
    const std::size_t size{static_cast<std::size_t>(st.range(0))};
    std::size_t i{0};

    while (st.KeepRunning()) {
      if (i == Max_Allocations) {
        st.PauseTiming();
        for (std::size_t j{0}; j < Max_Allocations; j++)
          deallocate(m_allocations[j]);
        i = 0;
        st.ResumeTiming();
      }
      m_allocations[i++] = allocate(size);
    }
    for (std::size_t j{0}; j < i; j++)
      deallocate(m_allocations[j]);
  }

  void deallocation(benchmark::State& st) {
    const std::size_t size{static_cast<std::size_t>(st.range(0))};
    std::size_t i{0};

    while (st.KeepRunning()) {
      if (i == 0 || i == Max_Allocations) {
        st.PauseTiming();
        for (std::size_t j{0}; j < Max_Allocations; j++)
          m_allocations[j] = allocate(size);
        i = 0;
        st.ResumeTiming();
      }
      deallocate(m_allocations[i++]);
    }
    for (std::size_t j{i}; j < Max_Allocations; j++)
      deallocate(m_allocations[j]);
  }

  void* m_allocations[Max_Allocations];
};

template <typename Alloc>
class ResourceAllocator : public AllocatorBenchmark
{
public:
  ResourceAllocator() : m_alloc{} {}
  virtual void* allocate(std::size_t nbytes) final { return m_alloc.allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc.deallocate(ptr); }
private:
  Alloc m_alloc;
};

class NoOpMalloc : public ResourceAllocator<umpire::alloc::NoOpAllocator> {};
BENCHMARK_DEFINE_F(NoOpMalloc, malloc)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(NoOpMalloc, free)(benchmark::State& st)   { deallocation(st); }

template <umpire::resource::MemoryResourceType Resource>
class MemoryResourceAllocator : public AllocatorBenchmark
{
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  MemoryResourceAllocator() : m_alloc{nullptr} {}

  void SetUp(benchmark::State&) override final {
    m_alloc = new umpire::Allocator{umpire::ResourceManager::getInstance().getAllocator(Resource)};
  }

  void TearDown(benchmark::State&) override final { delete m_alloc; }

  virtual void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }
private:
  umpire::Allocator* m_alloc;
};

class NoOpResource : public MemoryResourceAllocator<umpire::resource::NoOp> {};
BENCHMARK_DEFINE_F(NoOpResource, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(NoOpResource, deallocate)(benchmark::State& st) { deallocation(st); }

// Register all the benchmarks

// Base allocators
BENCHMARK_REGISTER_F(NoOpMalloc, malloc)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(NoOpMalloc, free)->Range(RangeLow, RangeHi);

// Resources
BENCHMARK_REGISTER_F(NoOpResource, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(NoOpResource, deallocate)->Range(RangeLow, RangeHi);

BENCHMARK_MAIN();
