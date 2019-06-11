//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <memory>
#include "benchmark/benchmark.h"
#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/FixedPool.hpp"

#include "umpire/util/FixedMallocPool.hpp"

class allocatorBenchmark : public ::benchmark::Fixture {
public:
  using ::benchmark::Fixture::SetUp;
  using ::benchmark::Fixture::TearDown;

  allocatorBenchmark() : max_allocations(100000) {
    allocations = new void*[max_allocations];
  }
  virtual ~allocatorBenchmark() {
    delete[] allocations;
  }

  virtual void* allocate( uint64_t nbytes ) = 0;
  virtual void deallocate( void* ptr ) = 0;

  void largeAllocDealloc(benchmark::State &st) {
    uint64_t size = (uint64_t)((uint64_t)st.range(0) * 1024 * 1024 * 1024);
    void* allocation;

    while (st.KeepRunning()) {
      allocation = allocate(size);
      deallocate(allocation);
    }
  }

  void allocation(benchmark::State &st) {
    uint64_t size = (uint64_t)st.range(0);
    uint64_t i = 0;

    while (st.KeepRunning()) {
      if ( i == max_allocations ) {
        st.PauseTiming();
        for (uint64_t j = 0; j < max_allocations; j++)
          deallocate(allocations[j]);
        i = 0;
        st.ResumeTiming();
      }
      allocations[i++] = allocate(size);
    }
    for (uint64_t j = 0; j < i; j++)
      deallocate(allocations[j]);
  }

  void deallocation(benchmark::State &st) {
    auto size = st.range(0);
    uint64_t i = 0;
    while (st.KeepRunning()) {
      if ( i == 0 || i == max_allocations ) {
        st.PauseTiming();
        for (uint64_t j = 0; j < max_allocations; j++)
          allocations[j] = allocate(size);
        i = 0;
        st.ResumeTiming();
      }
      deallocate(allocations[i++]);
    }
    for (uint64_t j = i; j < max_allocations; j++)
      deallocate(allocations[j]);
  }

  const uint64_t max_allocations;
  void** allocations;
};

class Malloc : public ::allocatorBenchmark {
  public:
  virtual void* allocate( uint64_t nbytes ) { return malloc(nbytes); }
  virtual void deallocate( void* ptr ) { free(ptr); }
};
BENCHMARK_DEFINE_F(Malloc, malloc)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(Malloc, free)(benchmark::State &st)   { deallocation(st); }

class allocator : public ::allocatorBenchmark {
public:
  using allocatorBenchmark::SetUp;
  using allocatorBenchmark::TearDown;
  void SetUp(const ::benchmark::State&) {
    auto& rm = umpire::ResourceManager::getInstance();
    allocator = new umpire::Allocator(rm.getAllocator(getName()));
  }
  void TearDown(const ::benchmark::State&) {
    delete allocator;
  }
  virtual void* allocate( uint64_t nbytes ) { return allocator->allocate(nbytes); }
  virtual void deallocate( void* ptr ) { allocator->deallocate(ptr); }
  virtual const std::string& getName( void ) = 0;

  umpire::Allocator* allocator;
};

class Host : public ::allocator {
  public:
    Host(): name("HOST") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(Host, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(Host, deallocate)(benchmark::State &st)   { deallocation(st); }

class Device : public ::allocator {
  public:
    Device(): name("DEVICE") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(Device, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(Device, deallocate)(benchmark::State &st)   { deallocation(st); }
BENCHMARK_DEFINE_F(Device, largeAllocDealloc)(benchmark::State &st)   { largeAllocDealloc(st); }

class UM : public ::allocator {
  public:
    UM(): name("UM") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(UM, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(UM, deallocate)(benchmark::State &st)   { deallocation(st); }
BENCHMARK_DEFINE_F(UM, largeAllocDealloc)(benchmark::State &st)   { largeAllocDealloc(st); }

static int namecnt = 0;   // Used to generate unique name per iteration
class Pool : public ::allocatorBenchmark {
public:
  using allocatorBenchmark::SetUp;
  using allocatorBenchmark::TearDown;
  void SetUp(const ::benchmark::State&) {
    std::stringstream ss;
    ss << "host_pool" << namecnt++;
    auto& rm = umpire::ResourceManager::getInstance();
    rm.makeAllocator<umpire::strategy::DynamicPool>(ss.str(), rm.getAllocator(getName()));
    allocator = new umpire::Allocator(rm.getAllocator(ss.str()));

    void* ptr;
    ptr = allocate(100);
    deallocate(ptr);
  }

  void TearDown(const ::benchmark::State&) {
    delete allocator;
  }
  virtual void* allocate( uint64_t nbytes ) { return allocator->allocate(nbytes); }
  virtual void deallocate( void* ptr ) { allocator->deallocate(ptr); }

  virtual const std::string& getName( void ) = 0;

  umpire::Allocator* allocator;
};

class PoolHost : public ::Pool {
  public:
    PoolHost(): name("HOST") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(PoolHost, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(PoolHost, deallocate)(benchmark::State &st)   { deallocation(st); }

class PoolDevice : public ::Pool {
  public:
    PoolDevice(): name("DEVICE") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(PoolDevice, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(PoolDevice, deallocate)(benchmark::State &st)   { deallocation(st); }

class PoolUM : public ::Pool {
  public:
    PoolUM(): name("UM") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(PoolUM, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(PoolUM, deallocate)(benchmark::State &st)   { deallocation(st); }

class FixedPool : public ::allocatorBenchmark {
public:
  using allocatorBenchmark::SetUp;
  using allocatorBenchmark::TearDown;

  void SetUp(const ::benchmark::State&) {
    std::stringstream ss;
    ss << "fixedpool" << namecnt++;
    auto& rm = umpire::ResourceManager::getInstance();
    rm.makeAllocator<umpire::strategy::FixedPool>(ss.str(), rm.getAllocator(getName()), 256, 128 * sizeof(int) * 8);
    allocator = new umpire::Allocator(rm.getAllocator(ss.str()));

    void* ptr;
    ptr = allocate(1);
    deallocate(ptr);
  }

  void TearDown(const ::benchmark::State&) {
    delete allocator;
  }

  virtual void* allocate( uint64_t nbytes ) { return allocator->allocate(nbytes); }
  virtual void deallocate( void* ptr ) { allocator->deallocate(ptr); }
  virtual const std::string& getName( void ) = 0;

  umpire::Allocator* allocator;
};

class FixedPoolHost : public ::FixedPool {
  public:
    FixedPoolHost(): name("HOST") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(FixedPoolHost, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolHost, deallocate)(benchmark::State &st)   { deallocation(st); }

class FixedPoolUM : public ::FixedPool {
  public:
    FixedPoolUM(): name("UM") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(FixedPoolUM, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolUM, deallocate)(benchmark::State &st)   { deallocation(st); }

class FixedPoolDevice : public ::FixedPool {
  public:
    FixedPoolDevice(): name("DEVICE") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(FixedPoolDevice, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedPoolDevice, deallocate)(benchmark::State &st)   { deallocation(st); }

class FixedMallocPool : public ::allocatorBenchmark {
  public:
  void SetUp(const ::benchmark::State&) override {
    pool = new umpire::util::FixedMallocPool(8);
  }
  void TearDown(const ::benchmark::State&) override {
    delete pool;
  }
  virtual void* allocate( uint64_t nbytes ) override final { return pool->allocate(nbytes); }
  virtual void deallocate( void* ptr ) override final { pool->deallocate(ptr); }
  private:
  umpire::util::FixedMallocPool *pool;
};

BENCHMARK_DEFINE_F(FixedMallocPool, allocate)(benchmark::State &st) { allocation(st); }
BENCHMARK_DEFINE_F(FixedMallocPool, deallocate)(benchmark::State &st)   { deallocation(st); }


static const int RangeLow = 4;
static const int RangeHi = 1024;

BENCHMARK_REGISTER_F(Malloc, malloc)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(Malloc, free)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedMallocPool, allocate)->Arg(RangeLow);
BENCHMARK_REGISTER_F(FixedMallocPool, deallocate)->Arg(RangeLow);
BENCHMARK_REGISTER_F(Host, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(Host, deallocate)->Range(RangeLow, RangeHi);
// BENCHMARK_REGISTER_F(PoolHost, allocate)->Range(RangeLow, RangeHi);
// BENCHMARK_REGISTER_F(PoolHost, deallocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(FixedPoolHost, allocate)->Arg(RangeLow);
BENCHMARK_REGISTER_F(FixedPoolHost, deallocate)->Arg(RangeLow);

#if defined(UMPIRE_ENABLE_CUDA)
BENCHMARK_REGISTER_F(Device, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(Device, deallocate)->Range(RangeLow, RangeHi);
//BENCHMARK_REGISTER_F(PoolDevice, allocate)->Range(RangeLow, RangeHi);
//BENCHMARK_REGISTER_F(PoolDevice, deallocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(UM, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(UM, deallocate)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(Device, largeAllocDealloc)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(9)->Arg(10)->Arg(11)->Arg(12)->Arg(13);
BENCHMARK_REGISTER_F(UM, largeAllocDealloc)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(9)->Arg(10)->Arg(11)->Arg(12)->Arg(13);

// BENCHMARK_REGISTER_F(PoolUM, allocate)->Range(RangeLow, RangeHi);
// BENCHMARK_REGISTER_F(PoolUM, deallocate)->Range(RangeLow, RangeHi);

BENCHMARK_REGISTER_F(FixedPoolDevice, allocate)->Arg(RangeLow);
BENCHMARK_REGISTER_F(FixedPoolDevice, deallocate)->Arg(RangeLow);
BENCHMARK_REGISTER_F(FixedPoolUM, allocate)->Arg(RangeLow);
BENCHMARK_REGISTER_F(FixedPoolUM, deallocate)->Arg(RangeLow);
#endif


BENCHMARK_MAIN()
