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

#include "umpire/alloc/MallocAllocator.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/alloc/CudaMallocAllocator.hpp"
#include "umpire/alloc/CudaMallocManagedAllocator.hpp"
#include "umpire/alloc/CudaPinnedAllocator.hpp"
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include "umpire/alloc/HipMallocAllocator.hpp"
#include "umpire/alloc/HipMallocManagedAllocator.hpp"
#include "umpire/alloc/HipPinnedAllocator.hpp"
#endif

#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/MixedPool.hpp"

#include "umpire/util/FixedMallocPool.hpp"

static const int RangeLow{16};
static const int RangeHi{64};

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

  void TearDown(benchmark::State&) override {
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

#if defined(UMPIRE_ENABLE_DEVICE)
class FixedPoolDevice : public FixedPool<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(FixedPoolDevice, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolDevice, deallocate)(benchmark::State& st) { deallocation(st); }

class FixedPoolDevicePinned : public FixedPool<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(FixedPoolDevicePinned, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolDevicePinned, deallocate)(benchmark::State& st) { deallocation(st); }
#endif

#if defined(UMPIRE_ENABLE_UM)
class FixedPoolUnified : public FixedPool<umpire::resource::Unified> {};
BENCHMARK_DEFINE_F(FixedPoolUnified, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolUnified, deallocate)(benchmark::State& st) { deallocation(st); }
#endif

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

#if defined(UMPIRE_ENABLE_DEVICE)
class DynamicPoolDevice : public DynamicPool<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(DynamicPoolDevice, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevice, deallocate)(benchmark::State& st) { deallocation(st); }

class DynamicPoolDevicePinned : public DynamicPool<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(DynamicPoolDevicePinned, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevicePinned, deallocate)(benchmark::State& st) { deallocation(st); }
#endif

#if defined(UMPIRE_ENABLE_UM)
class DynamicPoolUnified : public DynamicPool<umpire::resource::Unified> {};
BENCHMARK_DEFINE_F(DynamicPoolUnified, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolUnified, deallocate)(benchmark::State& st) { deallocation(st); }
#endif

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

#if defined(UMPIRE_ENABLE_DEVICE)
class MixedPoolDevice : public MixedPool<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(MixedPoolDevice, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(MixedPoolDevice, deallocate)(benchmark::State& st) { deallocation(st); }

class MixedPoolDevicePinned : public MixedPool<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(MixedPoolDevicePinned, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(MixedPoolDevicePinned, deallocate)(benchmark::State& st) { deallocation(st); }
#endif

#if defined(UMPIRE_ENABLE_UM)
class MixedPoolUnified : public MixedPool<umpire::resource::Unified> {};
BENCHMARK_DEFINE_F(MixedPoolUnified, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(MixedPoolUnified, deallocate)(benchmark::State& st) { deallocation(st); }
#endif

// Register all the benchmarks

// FixedMallocPool
BENCHMARK_REGISTER_F(FixedMallocPool, allocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedMallocPool, deallocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);

// FixedPool
BENCHMARK_REGISTER_F(FixedPoolHost, allocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolHost, deallocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);
#if defined(UMPIRE_ENABLE_DEVICE)
BENCHMARK_REGISTER_F(FixedPoolDevice, allocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolDevice, deallocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(FixedPoolDevicePinned, allocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolDevicePinned, deallocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);
#endif

#if defined(UMPIRE_ENABLE_UM)
BENCHMARK_REGISTER_F(FixedPoolUnified, allocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolUnified, deallocate)->RangeMultiplier(2)->Range(RangeLow, RangeHi);
#endif

// DynamicPool
BENCHMARK_REGISTER_F(DynamicPoolHost, allocate)->Args({16, 8192});
BENCHMARK_REGISTER_F(DynamicPoolHost, deallocate)->Args({16, 8192});
#if defined(UMPIRE_ENABLE_DEVICE)
BENCHMARK_REGISTER_F(DynamicPoolDevice, allocate)->Args({16, 8192});
BENCHMARK_REGISTER_F(DynamicPoolDevice, deallocate)->Args({16, 8192});

BENCHMARK_REGISTER_F(DynamicPoolDevicePinned, allocate)->Args({16, 8192});
BENCHMARK_REGISTER_F(DynamicPoolDevicePinned, deallocate)->Args({16, 8192});
#endif

#if defined(UMPIRE_ENABLE_UM)
BENCHMARK_REGISTER_F(DynamicPoolUnified, allocate)->Args({16, 8192});
BENCHMARK_REGISTER_F(DynamicPoolUnified, deallocate)->Args({16, 8192});
#endif

// MixedPool
BENCHMARK_REGISTER_F(MixedPoolHost, allocate)->Args({16, 8192});
BENCHMARK_REGISTER_F(MixedPoolHost, deallocate)->Args({16, 8192});
#if defined(UMPIRE_ENABLE_DEVICE)
BENCHMARK_REGISTER_F(MixedPoolDevice, allocate)->Args({16, 8192});
BENCHMARK_REGISTER_F(MixedPoolDevice, deallocate)->Args({16, 8192});

BENCHMARK_REGISTER_F(MixedPoolDevicePinned, allocate)->Args({16, 8192});
BENCHMARK_REGISTER_F(MixedPoolDevicePinned, deallocate)->Args({16, 8192});
#endif

#if defined(UMPIRE_ENABLE_UM)
BENCHMARK_REGISTER_F(MixedPoolUnified, allocate)->Args({16, 8192});
BENCHMARK_REGISTER_F(MixedPoolUnified, deallocate)->Args({16, 8192});
#endif


BENCHMARK_MAIN();
