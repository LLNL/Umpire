//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <memory>
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

static const std::size_t max_allocations = 100000;

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
      if (i == max_allocations) {
        st.PauseTiming();
        for (std::size_t j = 0; j < max_allocations; j++)
          deallocate(allocations[j]);
        i = 0;
        st.ResumeTiming();
      }
      allocations[i++] = allocate(size);
    }
    for (std::size_t j = 0; j < i; j++)
      deallocate(allocations[j]);
  }

  void deallocation(benchmark::State &st) {
    auto size = st.range(0);
    std::size_t i = 0;

    while (st.KeepRunning()) {
      if (i == 0 || i == max_allocations) {
        st.PauseTiming();
        for (std::size_t j = 0; j < max_allocations; j++)
          allocations[j] = allocate(size);
        i = 0;
        st.ResumeTiming();
      }
      deallocate(allocations[i++]);
    }
    for (std::size_t j = i; j < max_allocations; j++)
      deallocate(allocations[j]);
  }

  void* allocations[max_allocations];
};

template <typename Alloc>
class ResourceAllocator : public AllocatorBenchmark
{
public:
  ResourceAllocator() : m_alloc{} {}
  virtual void* allocate(std::size_t nbytes) { return m_alloc.allocate(nbytes); }
  virtual void deallocate(void* ptr) { m_alloc.deallocate(ptr); }
private:
  Alloc m_alloc;
};

class Malloc : public ResourceAllocator<umpire::alloc::MallocAllocator> {};
BENCHMARK_DEFINE_F(Malloc, malloc)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(Malloc, malloc)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(Malloc, free)(benchmark::State &st)   { deallocation(st); }
BENCHMARK_REGISTER_F(Malloc, free)->Range(RangeLow, RangeHi);

#if defined(UMPIRE_ENABLE_CUDA)
class CudaMalloc : public ResourceAllocator<umpire::alloc::CudaMallocAllocator> {};
BENCHMARK_DEFINE_F(CudaMalloc, cudaMalloc)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(CudaMalloc, cudaMalloc)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(CudaMalloc, cudaFree)(benchmark::State &st)   { deallocation(st); }
BENCHMARK_REGISTER_F(CudaMalloc, cudaFree)->Range(RangeLow, RangeHi);

class CudaMallocManaged : public ResourceAllocator<umpire::alloc::CudaMallocAllocator> {};
BENCHMARK_DEFINE_F(CudaMallocManaged, cudaMallocManaged)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(CudaMallocManaged, cudaMallocManaged)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(CudaMallocManaged, cudaFree)(benchmark::State &st)   { deallocation(st); }
BENCHMARK_REGISTER_F(CudaMallocManaged, cudaFree)->Range(RangeLow, RangeHi);

class CudaPinned : public ResourceAllocator<umpire::alloc::CudaPinnedAllocator> {};
BENCHMARK_DEFINE_F(CudaPinned, cudaMallocHost)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(CudaPinned, cudaMallocHost)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(CudaPinned, cudaFreeHost)(benchmark::State &st)   { deallocation(st); }
BENCHMARK_REGISTER_F(CudaPinned, cudaFreeHost)->Range(RangeLow, RangeHi);
#endif

#if defined(UMPIRE_ENABLE_HIP)
class HipMalloc : public ResourceAllocator<umpire::alloc::HipMallocAllocator> {};
BENCHMARK_DEFINE_F(HipMalloc, hipMalloc)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(HipMalloc, hipMalloc)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(HipMalloc, hipFree)(benchmark::State &st)   { deallocation(st); }
BENCHMARK_REGISTER_F(HipMalloc, hipFree)->Range(RangeLow, RangeHi);

class HipPinned : public ResourceAllocator<umpire::alloc::HipPinnedAllocator> {};
BENCHMARK_DEFINE_F(HipPinned, hipMallocHost)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(HipPinned, hipMallocHost)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(HipPinned, hipFreeHost)(benchmark::State &st)   { deallocation(st); }
BENCHMARK_REGISTER_F(HipPinned, hipFreeHost)->Range(RangeLow, RangeHi);
#endif

template <umpire::resource::MemoryResourceType Resource>
class MemoryResourceAllocator : public AllocatorBenchmark
{
public:
  MemoryResourceAllocator() :
    m_alloc{umpire::ResourceManager::getInstance().getAllocator(Resource)} {}
  virtual void* allocate(std::size_t nbytes) { return m_alloc.allocate(nbytes); }
  virtual void deallocate(void* ptr) { m_alloc.deallocate(ptr); }
private:
  umpire::Allocator m_alloc;
};

class HostResource : public MemoryResourceAllocator<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(HostResource, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(HostResource, allocate)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(HostResource, deallocate)(benchmark::State &st) { deallocation(st); }
BENCHMARK_REGISTER_F(HostResource, deallocate)->Range(RangeLow, RangeHi);

#if defined(UMPIRE_ENABLE_DEVICE)
class DeviceResource : public MemoryResourceAllocator<umpire::resource::Device> {};
BENCHMARK_DEFINE_F(DeviceResource, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(DeviceResource, allocate)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(DeviceResource, deallocate)(benchmark::State &st)   { deallocation(st); }
BENCHMARK_REGISTER_F(DeviceResource, deallocate)->Range(RangeLow, RangeHi);

class DeviceResource : public MemoryResourceAllocator<umpire::resource::Pinned> {};
BENCHMARK_DEFINE_F(DeviceResource, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(DeviceResource, allocate)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(DeviceResource, deallocate)(benchmark::State &st) { deallocation(st); }
BENCHMARK_REGISTER_F(DeviceResource, deallocate)->Range(RangeLow, RangeHi);
#endif

#if defined(UMPIRE_ENABLE_CUDA)
class UnifiedResource : public MemoryResourceAllocator<umpire::resource::Unified> {};
BENCHMARK_DEFINE_F(UnifiedResource, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(UnifiedResource, allocate)->Range(RangeLow, RangeHi);

BENCHMARK_DEFINE_F(UnifiedResource, deallocate)(benchmark::State &st) { deallocation(st); }
BENCHMARK_REGISTER_F(UnifiedResource, deallocate)->Range(RangeLow, RangeHi);
#endif

static int namecnt = 0;   // Used to generate unique name per iteration
template <umpire::resource::MemoryResourceType Resource>
class FixedPool : public AllocatorBenchmark {
public:
  using AllocatorBenchmark::SetUp;
  using AllocatorBenchmark::TearDown;

  FixedPool() : m_alloc{nullptr} {}

  void SetUp(const ::benchmark::State& st) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto bytes = static_cast<std::size_t>(st.range(0));

    std::stringstream ss;
    ss << "fixed_pool-" << Resource << "-" << bytes << "." << namecnt;
    ++namecnt;

    m_alloc = new umpire::Allocator{rm.makeAllocator<umpire::strategy::FixedPool>(
        ss.str(), rm.getAllocator(Resource), bytes, 128 * sizeof(int) * 8)};
  }

  void TearDown(const ::benchmark::State&) { delete m_alloc; }

  virtual void* allocate(std::size_t nbytes) { return m_alloc->allocate(nbytes); }
  virtual void deallocate(void* ptr) { m_alloc->deallocate(ptr); }

private:
  umpire::Allocator* m_alloc;
};

class FixedPoolHost : public FixedPool<umpire::resource::Host> {};
BENCHMARK_DEFINE_F(FixedPoolHost, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_REGISTER_F(FixedPoolHost, allocate)->Arg(256);

BENCHMARK_DEFINE_F(FixedPoolHost, deallocate)(benchmark::State &st) { deallocation(st); }
BENCHMARK_REGISTER_F(FixedPoolHost, deallocate)->Arg(256);

BENCHMARK_MAIN()
