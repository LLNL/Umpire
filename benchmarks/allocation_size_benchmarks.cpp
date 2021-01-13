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

class Malloc : public ResourceAllocator<umpire::alloc::MallocAllocator> {};
BENCHMARK_DEFINE_F(Malloc, malloc)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(Malloc, free)(benchmark::State& st)   { deallocation(st); }

#if defined(UMPIRE_ENABLE_CUDA)
class CudaMalloc : public ResourceAllocator<umpire::alloc::CudaMallocAllocator> {};
BENCHMARK_DEFINE_F(CudaMalloc, cudaMalloc)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(CudaMalloc, cudaFree)(benchmark::State& st)   { deallocation(st); }

class CudaMallocManaged : public ResourceAllocator<umpire::alloc::CudaMallocManagedAllocator> {};
BENCHMARK_DEFINE_F(CudaMallocManaged, cudaMallocManaged)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(CudaMallocManaged, cudaFree)(benchmark::State& st)   { deallocation(st); }

class CudaPinned : public ResourceAllocator<umpire::alloc::CudaPinnedAllocator> {};
BENCHMARK_DEFINE_F(CudaPinned, cudaMallocHost)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(CudaPinned, cudaFreeHost)(benchmark::State& st)   { deallocation(st); }
#endif

#if defined(UMPIRE_ENABLE_HIP)
class HipMalloc : public ResourceAllocator<umpire::alloc::HipMallocAllocator> {};
BENCHMARK_DEFINE_F(HipMalloc, hipMalloc)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(HipMalloc, hipFree)(benchmark::State& st)   { deallocation(st); }

class HipMallocManaged : public ResourceAllocator<umpire::alloc::HipMallocManagedAllocator> {};
BENCHMARK_DEFINE_F(HipMallocManaged, hipMallocManaged)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(HipMallocManaged, hipFree)(benchmark::State& st)   { deallocation(st); }

class HipPinned : public ResourceAllocator<umpire::alloc::HipPinnedAllocator> {};
BENCHMARK_DEFINE_F(HipPinned, hipMallocHost)(benchmark::State& st) { allocation(st); }
BENCHMARK_DEFINE_F(HipPinned, hipFreeHost)(benchmark::State& st)   { deallocation(st); }
#endif

// Register all the benchmarks
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

BENCHMARK_REGISTER_F(HipMallocManaged, hipMallocManaged)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(HipMallocManaged, hipFree)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(HipPinned, hipMallocHost)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(HipPinned, hipFreeHost)->Range(RangeLow, RangeHi);
#endif

BENCHMARK_MAIN();
