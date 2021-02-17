//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <random>
#include "benchmark/benchmark.h"

#include "umpire/config.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/MixedPool.hpp"

static const int RangeLow{1<<10}; //1kB
static const int RangeHi{1<<28}; //256MB
static const bool Introspection{false};

/*
 * Allocate either LARGE (about 12GB), MEDIUM (about 6GB)
 * or SMALL (about 1GB) for benchmark measurements.
 */
//#define LARGE 12000000000
//#define MEDIUM 6000000000
#define SMALL 1000000000

class AllocatorBenchmark : public benchmark::Fixture {
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  virtual void* allocate(std::size_t nbytes) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual void release() = 0;

  void allocation(benchmark::State& st) {
    const std::size_t size{static_cast<std::size_t>(st.range(0))};
    std::size_t i{0};

    Max_Allocations = setBounds(size);
    m_allocations.resize(Max_Allocations);

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

    // This says that we process with the rate of state.range(0) bytes every iteration:
    st.counters["BytesProcessed"] = benchmark::Counter(st.range(0), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

    for (std::size_t j{0}; j < i; j++)
      deallocate(m_allocations[j]);
    
    release();
  }

  void deallocation_same_order(benchmark::State& st) {
    const std::size_t size{static_cast<std::size_t>(st.range(0))};
    std::size_t i{0};

    Max_Allocations = setBounds(size);
    m_allocations.resize(Max_Allocations);

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

    // This says that we process with the rate of state.range(0) bytes every iteration:
    st.counters["BytesProcessed"] = benchmark::Counter(st.range(0), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

    for (std::size_t j{i}; j < Max_Allocations; j++)
      deallocate(m_allocations[j]);
    
    release();
  }

  void deallocation_reverse_order(benchmark::State& st) {
    const std::size_t size{static_cast<std::size_t>(st.range(0))};
    
    Max_Allocations = setBounds(size);
    int i{int(Max_Allocations)};
    m_allocations.resize(Max_Allocations);

    while (st.KeepRunning()) {
      if (i == 0 || i == int(Max_Allocations)) {
        st.PauseTiming();
        for (int j{0}; j < int(Max_Allocations); j++) 
          m_allocations[j] = allocate(size);
        i = Max_Allocations;
        st.ResumeTiming();
      }
      deallocate(m_allocations[--i]);
    }

    // This says that we process with the rate of state.range(0) bytes every iteration:
    st.counters["BytesProcessed"] = benchmark::Counter(st.range(0), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

    //for (int j{i}; j < int(Max_Allocations); j++)
    //  deallocate(m_allocations[j]);
    
    release();
  }

  void deallocation_random_order(benchmark::State& st) {
    std::mt19937 gen(Max_Allocations);
    const std::size_t size{static_cast<std::size_t>(st.range(0))};
    std::size_t i{0};
     
    Max_Allocations = setBounds(size);
    m_allocations.resize(Max_Allocations);

    while (st.KeepRunning()) {
      if (i == 0 || i == Max_Allocations) {
        st.PauseTiming();
        for (std::size_t j{0}; j < Max_Allocations; j++)
          m_allocations[j] = allocate(size);
        i = 0;
        std::shuffle(&m_allocations[0], &m_allocations[Max_Allocations], gen);
        st.ResumeTiming();
      }
      deallocate(m_allocations[i++]);
    }

    // This says that we process with the rate of state.range(0) bytes every iteration:
    st.counters["BytesProcessed"] = benchmark::Counter(st.range(0), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);

    for (std::size_t j{i}; j < Max_Allocations; j++)
      deallocate(m_allocations[j]);
    
    release();
  }

  /*
  * This function figures out, given the RangeHi and RangeLo,
  * what the value of Max_Allocations should be. The goal is
  * to allocate more with a smaller range and less with a large
  * range, while keeping the total bytes allocated about the same.
  */
  std::size_t setBounds(std::size_t size) {
    #if defined(LARGE)
      return(LARGE/size);
    #elif defined(MEDIUM)
      return(MEDIUM/size);
    #else
      return(SMALL/size);
    #endif
  }

  std::vector<void*> m_allocations;
  std::size_t Max_Allocations;
};

/*
 * FixedPool
 */
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
        ss.str(), rm.getAllocator(Resource), bytes)};
  }

  void TearDown(benchmark::State&) override {
    m_alloc->getAllocationStrategy()->release();
    delete m_alloc;
  }

  virtual void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }
  virtual void release() final { m_alloc->getAllocationStrategy()->release(); }

private:
  umpire::Allocator* m_alloc;
};

class FixedPoolHost : public FixedPool<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(FixedPoolHost, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolHost, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(FixedPoolHost, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(FixedPoolHost, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }

#if defined(UMPIRE_ENABLE_DEVICE)
class FixedPoolDevice : public FixedPool<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(FixedPoolDevice, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolDevice, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(FixedPoolDevice, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(FixedPoolDevice, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }

class FixedPoolDevicePinned : public FixedPool<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(FixedPoolDevicePinned, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolDevicePinned, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(FixedPoolDevicePinned, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(FixedPoolDevicePinned, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }
#endif

#if defined(UMPIRE_ENABLE_UM)
class FixedPoolUnified : public FixedPool<umpire::resource::Unified> {};
BENCHMARK_DEFINE_F(FixedPoolUnified, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolUnified, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(FixedPoolUnified, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(FixedPoolUnified, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }
#endif

/*
 * DynamicPool
 */
template <umpire::resource::MemoryResourceType Resource>
class DynamicPool : public AllocatorBenchmark {
public:
  using AllocatorBenchmark::SetUp;
  using AllocatorBenchmark::TearDown;

  DynamicPool() : m_alloc{nullptr} {}

  void SetUp(benchmark::State&) override final{
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
  virtual void release() final { m_alloc->release(); }

private:
  umpire::Allocator* m_alloc;
};

class DynamicPoolHost : public DynamicPool<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(DynamicPoolHost, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolHost, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(DynamicPoolHost, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(DynamicPoolHost, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }

#if defined(UMPIRE_ENABLE_DEVICE)
class DynamicPoolDevice : public DynamicPool<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(DynamicPoolDevice, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevice, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevice, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevice, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }

class DynamicPoolDevicePinned : public DynamicPool<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(DynamicPoolDevicePinned, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevicePinned, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevicePinned, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevicePinned, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }
#endif

#if defined(UMPIRE_ENABLE_UM)
class DynamicPoolUnified : public DynamicPool<umpire::resource::Unified> {};
BENCHMARK_DEFINE_F(DynamicPoolUnified, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolUnified, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(DynamicPoolUnified, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(DynamicPoolUnified, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }
#endif

/*
 * MixedPool
 */
template <umpire::resource::MemoryResourceType Resource>
class MixedPool : public AllocatorBenchmark {
public:
  using AllocatorBenchmark::SetUp;
  using AllocatorBenchmark::TearDown;

  MixedPool() : m_alloc{nullptr} {}

  void SetUp(benchmark::State&) override final{
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
  virtual void release() final { m_alloc->release(); }

private:
  umpire::Allocator* m_alloc;
};

class MixedPoolHost : public MixedPool<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(MixedPoolHost, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(MixedPoolHost, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(MixedPoolHost, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(MixedPoolHost, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }

#if defined(UMPIRE_ENABLE_DEVICE)
class MixedPoolDevice : public MixedPool<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(MixedPoolDevice, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(MixedPoolDevice, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(MixedPoolDevice, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(MixedPoolDevice, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }

class MixedPoolDevicePinned : public MixedPool<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(MixedPoolDevicePinned, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(MixedPoolDevicePinned, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(MixedPoolDevicePinned, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(MixedPoolDevicePinned, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }
#endif

#if defined(UMPIRE_ENABLE_UM)
class MixedPoolUnified : public MixedPool<umpire::resource::Unified> {};
BENCHMARK_DEFINE_F(MixedPoolUnified, allocate)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(MixedPoolUnified, deallocate_same_order)(benchmark::State& st) { deallocation_same_order(st); }
BENCHMARK_DEFINE_F(MixedPoolUnified, deallocate_reverse_order)(benchmark::State& st) { deallocation_reverse_order(st); }
BENCHMARK_DEFINE_F(MixedPoolUnified, deallocate_random_order)(benchmark::State& st) { deallocation_random_order(st); }
#endif

// Register all the benchmarks

// FixedPool
BENCHMARK_REGISTER_F(FixedPoolHost, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolHost, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolHost, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolHost, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
#if defined(UMPIRE_ENABLE_DEVICE)
BENCHMARK_REGISTER_F(FixedPoolDevice, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolDevice, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolDevice, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolDevice, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(FixedPoolDevicePinned, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolDevicePinned, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolDevicePinned, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolDevicePinned, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
#endif

#if defined(UMPIRE_ENABLE_UM)
BENCHMARK_REGISTER_F(FixedPoolUnified, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolUnified, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolUnified, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolUnified, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
#endif

// DynamicPool
BENCHMARK_REGISTER_F(DynamicPoolHost, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolHost, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolHost, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolHost, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
#if defined(UMPIRE_ENABLE_DEVICE)
BENCHMARK_REGISTER_F(DynamicPoolDevice, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolDevice, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolDevice, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolDevice, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(DynamicPoolDevicePinned, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolDevicePinned, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolDevicePinned, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolDevicePinned, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
#endif

#if defined(UMPIRE_ENABLE_UM)
BENCHMARK_REGISTER_F(DynamicPoolUnified, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolUnified, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolUnified, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DynamicPoolUnified, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
#endif

// MixedPool
BENCHMARK_REGISTER_F(MixedPoolHost, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolHost, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolHost, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolHost, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
#if defined(UMPIRE_ENABLE_DEVICE)
BENCHMARK_REGISTER_F(MixedPoolDevice, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolDevice, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolDevice, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolDevice, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(MixedPoolDevicePinned, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolDevicePinned, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolDevicePinned, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolDevicePinned, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
#endif

#if defined(UMPIRE_ENABLE_UM)
BENCHMARK_REGISTER_F(MixedPoolUnified, allocate)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolUnified, deallocate_same_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolUnified, deallocate_reverse_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(MixedPoolUnified, deallocate_random_order)->RangeMultiplier(4)->Range(RangeLow, RangeHi);
#endif


BENCHMARK_MAIN();
