//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
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
#include "umpire/alloc/CudaPinnedAllocator.hpp"
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include "umpire/alloc/HipMallocAllocator.hpp"
#include "umpire/alloc/HipPinnedAllocator.hpp"
#endif

#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/FixedPool.hpp"

#include "umpire/util/FixedMallocPool.hpp"

static const int RangeLow = 4;
static const int RangeHi = 1024;

static const bool Introspection = true;

static const std::size_t Max_Allocations = 100000;
static const std::size_t Num_Random = 1000;

class AllocatorBenchmark : public benchmark::Fixture {
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  virtual void* allocate(std::size_t nbytes) = 0;
  virtual void deallocate(void* ptr) = 0;

  void largeAllocDealloc(benchmark::State &st) {
    std::size_t size = static_cast<std::size_t>(st.range(0)) * 1024 * 1024 * 1024;
    void* allocation;

    while (st.KeepRunning()) {
      allocation = allocate(size);
      deallocate(allocation);
    }
  }

  void allocation(benchmark::State &st) {
    std::size_t size = static_cast<std::size_t>(st.range(0));
    std::size_t i = 0;

    while (st.KeepRunning()) {
      if (i == Max_Allocations) {
        st.PauseTiming();
        for (std::size_t j = 0; j < Max_Allocations; j++)
          deallocate(m_allocations[j]);
        i = 0;
        st.ResumeTiming();
      }
      m_allocations[i++] = allocate(size);
    }
    for (std::size_t j = 0; j < i; j++)
      deallocate(m_allocations[j]);
  }

  void deallocation(benchmark::State &st) {
    auto size = st.range(0);
    std::size_t i = 0;

    while (st.KeepRunning()) {
      if (i == 0 || i == Max_Allocations) {
        st.PauseTiming();
        for (std::size_t j = 0; j < Max_Allocations; j++)
          m_allocations[j] = allocate(size);
        i = 0;
        st.ResumeTiming();
      }
      deallocate(m_allocations[i++]);
    }
    for (std::size_t j = i; j < Max_Allocations; j++)
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

class Malloc : public ResourceAllocator<umpire::alloc::MallocAllocator> {};
BENCHMARK_DEFINE_F(Malloc, malloc)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(Malloc, free)(benchmark::State &st)   { deallocation(st); }

#if defined(UMPIRE_ENABLE_CUDA)
class CudaMalloc : public ResourceAllocator<umpire::alloc::CudaMallocAllocator> {};
BENCHMARK_DEFINE_F(CudaMalloc, cudaMalloc)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(CudaMalloc, cudaFree)(benchmark::State &st)   { deallocation(st); }

class CudaMallocManaged : public ResourceAllocator<umpire::alloc::CudaMallocAllocator> {};
BENCHMARK_DEFINE_F(CudaMallocManaged, cudaMallocManaged)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(CudaMallocManaged, cudaFree)(benchmark::State &st)   { deallocation(st); }

class CudaPinned : public ResourceAllocator<umpire::alloc::CudaPinnedAllocator> {};
BENCHMARK_DEFINE_F(CudaPinned, cudaMallocHost)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(CudaPinned, cudaFreeHost)(benchmark::State &st)   { deallocation(st); }
#endif

#if defined(UMPIRE_ENABLE_HIP)
class HipMalloc : public ResourceAllocator<umpire::alloc::HipMallocAllocator> {};
BENCHMARK_DEFINE_F(HipMalloc, hipMalloc)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(HipMalloc, hipFree)(benchmark::State &st)   { deallocation(st); }

class HipPinned : public ResourceAllocator<umpire::alloc::HipPinnedAllocator> {};
BENCHMARK_DEFINE_F(HipPinned, hipMallocHost)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(HipPinned, hipFreeHost)(benchmark::State &st)   { deallocation(st); }
#endif

template <umpire::resource::MemoryResourceType Resource>
class MemoryResourceAllocator : public AllocatorBenchmark
{
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  MemoryResourceAllocator() = default;

  void SetUp(const ::benchmark::State&) override final {
    m_alloc = new umpire::Allocator{umpire::ResourceManager::getInstance().getAllocator(Resource)};
  }

  void TearDown(const ::benchmark::State&) override final { delete m_alloc; }

  virtual void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }
private:
  umpire::Allocator* m_alloc;
};

class HostResource : public MemoryResourceAllocator<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(HostResource, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(HostResource, deallocate)(benchmark::State &st) { deallocation(st); }

#if defined(UMPIRE_ENABLE_DEVICE)
class DeviceResource : public MemoryResourceAllocator<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(DeviceResource, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(DeviceResource, deallocate)(benchmark::State &st)   { deallocation(st); }

class DevicePinnedResource : public MemoryResourceAllocator<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(DevicePinnedResource, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(DevicePinnedResource, deallocate)(benchmark::State &st) { deallocation(st); }
#endif

#if defined(UMPIRE_ENABLE_CUDA)
class UnifiedResource : public MemoryResourceAllocator<umpire::resource::Unified> {};
BENCHMARK_DEFINE_F(UnifiedResource, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(UnifiedResource, deallocate)(benchmark::State &st) { deallocation(st); }
#endif

class FixedMallocPool : public AllocatorBenchmark {
public:
  using AllocatorBenchmark::SetUp;
  using AllocatorBenchmark::TearDown;

  void SetUp(const ::benchmark::State&) override final {
    m_alloc = new umpire::util::FixedMallocPool(8);
  }
  void TearDown(const ::benchmark::State&) override final {
    delete m_alloc;
  }

  virtual void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }

private:
  umpire::util::FixedMallocPool* m_alloc;
};

BENCHMARK_DEFINE_F(FixedMallocPool, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedMallocPool, deallocate)(benchmark::State &st) { deallocation(st); }

static int namecnt = 0;   // Used to generate unique name per iteration
template <umpire::resource::MemoryResourceType Resource>
class FixedPool : public AllocatorBenchmark {
public:
  using AllocatorBenchmark::SetUp;
  using AllocatorBenchmark::TearDown;

  FixedPool() : m_alloc{nullptr} {}

  void SetUp(const ::benchmark::State& st) override final {
    auto& rm = umpire::ResourceManager::getInstance();
    auto bytes = static_cast<std::size_t>(st.range(0));

    std::stringstream ss;
    ss << "fixed_pool-" << Resource << "-" << bytes << "." << namecnt;
    ++namecnt;

    m_alloc = new umpire::Allocator{rm.makeAllocator<umpire::strategy::FixedPool, Introspection>(
        ss.str(), rm.getAllocator(Resource), bytes, 128 * sizeof(int) * 8)};
  }

  void TearDown(const ::benchmark::State&) override final { delete m_alloc; }

  virtual void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }

private:
  umpire::Allocator* m_alloc;
};

class FixedPoolHost : public FixedPool<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(FixedPoolHost, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolHost, deallocate)(benchmark::State &st) { deallocation(st); }

#if defined(UMPIRE_ENABLE_DEVICE)
class FixedPoolDevice : public FixedPool<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(FixedPoolDevice, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolDevice, deallocate)(benchmark::State &st) { deallocation(st); }

class FixedPoolDevicePinned : public FixedPool<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(FixedPoolDevicePinned, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolDevicePinned, deallocate)(benchmark::State &st) { deallocation(st); }
#endif

#if defined(UMPIRE_ENABLE_CUDA)
class FixedPoolUnified : public FixedPool<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(FixedPoolUnified, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolUnified, deallocate)(benchmark::State &st) { deallocation(st); }
#endif

class AllocatorRandomSizeBenchmark : public benchmark::Fixture {
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  void SetUp(benchmark::State& st) {
    const std::size_t range_lo = static_cast<int>(st.range(0));
    const std::size_t range_hi = static_cast<int>(st.range(1));

    std::default_random_engine generator;
    generator.seed(0);

    std::uniform_int_distribution<std::size_t> distribution{range_lo, range_hi};

    auto random_number = std::bind(distribution, generator);

    std::generate(m_bytes, m_bytes + Num_Random,
                  [&random_number] () { return random_number(); });

    setUpPool();
  }

  virtual void* allocate(std::size_t nbytes) = 0;
  virtual void deallocate(void* ptr) = 0;
  virtual void setUpPool() = 0;

  void allocation(benchmark::State &st) {
    std::size_t i = 0;

    while (st.KeepRunning()) {
      if (i == Max_Allocations) {
        st.PauseTiming();
        for (std::size_t j = 0; j < Max_Allocations; j++) {
          deallocate(m_allocations[j]);
        }
        i = 0;
        st.ResumeTiming();
      }
      {
        m_allocations[i] = allocate(m_bytes[i % Num_Random]);
      }
      ++i;
    }
    for (std::size_t j = 0; j < i; j++) {
      deallocate(m_allocations[j]);
    }
  }

  void deallocation(benchmark::State &st) {
    std::size_t i = 0;

    while (st.KeepRunning()) {
      if (i == 0 || i == Max_Allocations) {
        st.PauseTiming();
        for (std::size_t j = 0; j < Max_Allocations; j++) {
          m_allocations[j] = allocate(m_bytes[j % Num_Random]);
        }
        i = 0;
        st.ResumeTiming();
      }
      {
        deallocate(m_allocations[i]);
      }
      ++i;
    }
    for (std::size_t j = i; j < Max_Allocations; j++) {
      deallocate(m_allocations[j]);
    }
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

  void TearDown(const ::benchmark::State&) override final { delete m_alloc; }

  virtual void setUpPool() final {
    auto& rm = umpire::ResourceManager::getInstance();

    std::stringstream ss;
    ss << "dynamic_pool-" << Resource << "." << namecnt;
    ++namecnt;

    m_alloc = new umpire::Allocator{rm.makeAllocator<umpire::strategy::DynamicPool, Introspection>(
        ss.str(), rm.getAllocator(Resource))};
  }

  virtual void* allocate(std::size_t nbytes) final { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) final { m_alloc->deallocate(ptr); }

private:
  umpire::Allocator* m_alloc;
};

class DynamicPoolHost : public DynamicPool<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(DynamicPoolHost, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolHost, deallocate)(benchmark::State &st) { deallocation(st); }

#if defined(UMPIRE_ENABLE_DEVICE)
class DynamicPoolDevice : public DynamicPool<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(DynamicPoolDevice, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevice, deallocate)(benchmark::State &st) { deallocation(st); }

class DynamicPoolDevicePinned : public DynamicPool<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(DynamicPoolDevicePinned, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolDevicePinned, deallocate)(benchmark::State &st) { deallocation(st); }
#endif

#if defined(UMPIRE_ENABLE_CUDA)
class DynamicPoolUnified : public DynamicPool<umpire::resource::Unified> {};
BENCHMARK_DEFINE_F(DynamicPoolUnified, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(DynamicPoolUnified, deallocate)(benchmark::State &st) { deallocation(st); }
#endif

// Register all the benchmarks

// Base allocators
BENCHMARK_REGISTER_F(Malloc, malloc)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(Malloc, free)->Range(RangeLow, RangeHi);

#if defined(UMPIRE_ENABLE_CUDA)
BENCHMARK_REGISTER_F(CudaMalloc, cudaMalloc)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(CudaMalloc, cudaFree)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(CudaMallocManaged, cudaMallocManaged)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(CudaMallocManaged, cudaFree)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(CudaPinned, cudaMallocHost)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(CudaPinned, cudaFreeHost)->Range(RangeLow, RangeHi);
#endif

#if defined(UMPIRE_ENABLE_HIP)
BENCHMARK_REGISTER_F(HipMalloc, hipMalloc)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(HipMalloc, hipFree)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(HipPinned, hipMallocHost)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(HipPinned, hipFreeHost)->Range(RangeLow, RangeHi);
#endif

// Resources
BENCHMARK_REGISTER_F(HostResource, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(HostResource, deallocate)->Range(RangeLow, RangeHi);

#if defined(UMPIRE_ENABLE_DEVICE)
BENCHMARK_REGISTER_F(DeviceResource, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DeviceResource, deallocate)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(DevicePinnedResource, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(DevicePinnedResource, deallocate)->Range(RangeLow, RangeHi);
#endif

#if defined(UMPIRE_ENABLE_CUDA)
BENCHMARK_REGISTER_F(UnifiedResource, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(UnifiedResource, deallocate)->Range(RangeLow, RangeHi);
#endif

// Pools

// FixedPool
BENCHMARK_REGISTER_F(FixedPoolHost, allocate)->Arg(256);
BENCHMARK_REGISTER_F(FixedPoolHost, deallocate)->Arg(256);
#if defined(UMPIRE_ENABLE_DEVICE)
BENCHMARK_REGISTER_F(FixedPoolDevice, allocate)->Arg(256);
BENCHMARK_REGISTER_F(FixedPoolDevice, deallocate)->Arg(256);

BENCHMARK_REGISTER_F(FixedPoolDevicePinned, allocate)->Arg(256);
BENCHMARK_REGISTER_F(FixedPoolDevicePinned, deallocate)->Arg(256);
#endif

#if defined(UMPIRE_ENABLE_CUDA)
BENCHMARK_REGISTER_F(FixedPoolUnified, allocate)->Arg(256);
BENCHMARK_REGISTER_F(FixedPoolUnified, deallocate)->Arg(256);
#endif

// DynamicPool
BENCHMARK_REGISTER_F(DynamicPoolHost, allocate)->Args({16, 1024});
BENCHMARK_REGISTER_F(DynamicPoolHost, deallocate)->Args({16, 1024});
#if defined(UMPIRE_ENABLE_DEVICE)
BENCHMARK_REGISTER_F(DynamicPoolDevice, allocate)->Args({16, 1024});
BENCHMARK_REGISTER_F(DynamicPoolDevice, deallocate)->Args({16, 1024});

BENCHMARK_REGISTER_F(DynamicPoolDevicePinned, allocate)->Args({16, 1024});
BENCHMARK_REGISTER_F(DynamicPoolDevicePinned, deallocate)->Args({16, 1024});
#endif

#if defined(UMPIRE_ENABLE_CUDA)
BENCHMARK_REGISTER_F(DynamicPoolUnified, allocate)->Args({16, 1024});
BENCHMARK_REGISTER_F(DynamicPoolUnified, deallocate)->Args({16, 1024});
#endif


BENCHMARK_MAIN()
