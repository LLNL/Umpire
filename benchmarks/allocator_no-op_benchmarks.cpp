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

#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/MixedPool.hpp"

#include "umpire/util/FixedMallocPool.hpp"

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

class HostResource : public MemoryResourceAllocator<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(HostResource, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(HostResource, deallocate)(benchmark::State& st) { deallocation(st); }

class FixedMallocPool : public AllocatorBenchmark {
public:
  using AllocatorBenchmark::SetUp;
  using AllocatorBenchmark::TearDown;

  FixedMallocPool() : m_alloc{nullptr} {}

  void SetUp(benchmark::State& st) override final {
    const std::size_t bytes{static_cast<std::size_t>(st.range(0))};
    m_alloc = new umpire::util::FixedMallocPool(bytes);
  }
  void TearDown(benchmark::State&) override final {
    delete m_alloc;
  }

  virtual void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }

private:
  umpire::util::FixedMallocPool* m_alloc;
};

BENCHMARK_DEFINE_F(FixedMallocPool, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedMallocPool, deallocate)(benchmark::State& st) { deallocation(st); }

static int namecnt = 0;   // Used to generate unique name per iteration
template <umpire::resource::MemoryResourceType Resource>
class FixedPool : public AllocatorBenchmark {
public:
  using AllocatorBenchmark::SetUp;
  using AllocatorBenchmark::TearDown;

  FixedPool() : m_alloc{nullptr} {}

  void SetUp(benchmark::State& st) override final {
    auto& rm = umpire::ResourceManager::getInstance();
    const std::size_t bytes{static_cast<std::size_t>(st.range(0))};

    std::stringstream ss;
    ss << "fixed_pool-" << Resource << "-" << bytes << "." << namecnt;
    ++namecnt;

    m_alloc = new umpire::Allocator{rm.makeAllocator<umpire::strategy::FixedPool, Introspection>(
        ss.str(), rm.getAllocator(Resource), bytes, 128 * sizeof(int) * 8)};
  }

  void TearDown(benchmark::State&) override final {
    m_alloc->getAllocationStrategy()->release();
    delete m_alloc;
  }

  virtual void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }

private:
  umpire::Allocator* m_alloc;
};

class FixedPoolHost : public FixedPool<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(FixedPoolHost, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolHost, deallocate)(benchmark::State& st) { deallocation(st); }

class AllocatorRandomSizeBenchmark : public benchmark::Fixture {
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  virtual void SetUpPool() = 0;

  void SetUp(benchmark::State& st) {
    const std::size_t range_lo{static_cast<std::size_t>(st.range(0))};
    const std::size_t range_hi{static_cast<std::size_t>(st.range(1))};

    std::default_random_engine generator;
    generator.seed(0);

    std::uniform_int_distribution<std::size_t> distribution{range_lo, range_hi};

    auto random_number = std::bind(distribution, generator);

    std::generate(m_bytes, m_bytes + Num_Random,
                  [&random_number] () { return random_number(); });

    SetUpPool();
  }

  virtual void* allocate(std::size_t nbytes) = 0;
  virtual void deallocate(void* ptr) = 0;

  void allocation(benchmark::State& st) {
    std::size_t i{0};

    while (st.KeepRunning()) {
      if (i == Max_Allocations) {
        st.PauseTiming();
        for (std::size_t j{0}; j < Max_Allocations; j++) {
          deallocate(m_allocations[j]);
        }
        i = 0;
        st.ResumeTiming();
      }
      m_allocations[i] = allocate(m_bytes[i % Num_Random]);
      ++i;
    }
    for (std::size_t j{0}; j < i; j++)
      deallocate(m_allocations[j]);
  }

  void deallocation(benchmark::State& st) {
    std::size_t i{0};

    while (st.KeepRunning()) {
      if (i == 0 || i == Max_Allocations) {
        st.PauseTiming();
        for (std::size_t j{0}; j < Max_Allocations; j++) {
          m_allocations[j] = allocate(m_bytes[j % Num_Random]);
        }
        i = 0;
        st.ResumeTiming();
      }
      deallocate(m_allocations[i]);
      ++i;
    }
    for (std::size_t j{i}; j < Max_Allocations; j++)
      deallocate(m_allocations[j]);
  }

  void* m_allocations[Max_Allocations];
  std::size_t m_bytes[Num_Random];
};

template <umpire::resource::MemoryResourceType Resource>
class DynamicPool : public AllocatorRandomSizeBenchmark {
public:
  using AllocatorRandomSizeBenchmark::SetUp;
  using AllocatorRandomSizeBenchmark::TearDown;

  DynamicPool() : m_alloc{nullptr} {}

  void SetUpPool() final {
    auto& rm = umpire::ResourceManager::getInstance();

    std::stringstream ss;
    ss << "dynamic_pool-" << Resource << "." << namecnt;
    ++namecnt;

    m_alloc = new umpire::Allocator{
      rm.makeAllocator<umpire::strategy::DynamicPool, Introspection>(
        ss.str(), rm.getAllocator(Resource))};
  }

  void TearDown(benchmark::State&) override final {
    m_alloc->getAllocationStrategy()->release();
    delete m_alloc;
  }

  virtual void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }

private:
  umpire::Allocator* m_alloc;
};

class DynamicPoolHost : public DynamicPool<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(DynamicPoolHost, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolHost, deallocate)(benchmark::State& st) { deallocation(st); }

template <umpire::resource::MemoryResourceType Resource>
class MixedPool : public AllocatorRandomSizeBenchmark {
public:
  using AllocatorRandomSizeBenchmark::SetUp;
  using AllocatorRandomSizeBenchmark::TearDown;

  MixedPool() : m_alloc{nullptr} {}

  void SetUpPool() override {
    auto& rm = umpire::ResourceManager::getInstance();

    std::stringstream ss;
    ss << "mixed_pool-" << Resource << "." << namecnt;
    ++namecnt;

    m_alloc = new umpire::Allocator{rm.makeAllocator<umpire::strategy::MixedPool, Introspection>(
        ss.str(), rm.getAllocator(Resource))};
  }

  void TearDown(benchmark::State&) override final {
    m_alloc->getAllocationStrategy()->release();
    delete m_alloc;
  }

  void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }

private:
  umpire::Allocator* m_alloc;
};

class MixedPoolHost : public MixedPool<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(MixedPoolHost, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(MixedPoolHost, deallocate)(benchmark::State& st) { deallocation(st); }

// Register all the benchmarks

// Base allocators
BENCHMARK_REGISTER_F(NoOpMalloc, malloc)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(NoOpMalloc, free)->Range(RangeLow, RangeHi);

// Resources
BENCHMARK_REGISTER_F(HostResource, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(HostResource, deallocate)->Range(RangeLow, RangeHi);

// Pools

// FixedMallocPool
BENCHMARK_REGISTER_F(FixedMallocPool, allocate)->Arg(256);
BENCHMARK_REGISTER_F(FixedMallocPool, deallocate)->Arg(256);

// FixedPool
BENCHMARK_REGISTER_F(FixedPoolHost, allocate)->Arg(256);
BENCHMARK_REGISTER_F(FixedPoolHost, deallocate)->Arg(256);

// DynamicPool
BENCHMARK_REGISTER_F(DynamicPoolHost, allocate)->Args({16, 1024});
BENCHMARK_REGISTER_F(DynamicPoolHost, deallocate)->Args({16, 1024});

// MixedPool
BENCHMARK_REGISTER_F(MixedPoolHost, allocate)->Args({16, 1024});
BENCHMARK_REGISTER_F(MixedPoolHost, deallocate)->Args({16, 1024});

BENCHMARK_MAIN();
