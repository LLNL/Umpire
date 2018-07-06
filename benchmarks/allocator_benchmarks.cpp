//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

class allocatorBenchmark : public ::benchmark::Fixture {
  public:
  allocatorBenchmark() : max_allocations(100000) { 
    allocations = new void*[max_allocations];
  }
  virtual ~allocatorBenchmark() {
    delete[] allocations;
  }
  virtual void* allocate( size_t nbytes ) = 0;
  virtual void deallocate( void* ptr ) = 0;
  void BM_allocation(benchmark::State &st) {
    auto size = st.range(0);
    size_t i = 0;
    while (st.KeepRunning()) {
      if ( i == max_allocations ) {
        st.PauseTiming();
        for (size_t j = 0; j < max_allocations; j++)
          deallocate(allocations[j]);
        i = 0;
        st.ResumeTiming();
      }
      allocations[i++] = allocate(size);
    }
    for (size_t j = 0; j < i; j++)
      deallocate(allocations[j]);
  }
  void BM_deallocation(benchmark::State &st) {
    auto size = st.range(0);
    size_t i = 0;
    while (st.KeepRunning()) {
      if ( i == 0 || i == max_allocations ) {
        st.PauseTiming();
        for (size_t j = 0; j < max_allocations; j++)
          allocations[j] = allocate(size);
        i = 0;
        st.ResumeTiming();
      }
      deallocate(allocations[i++]);
    }
    for (size_t j = i; j < max_allocations; j++)
      deallocate(allocations[j]);
  }

  const size_t max_allocations;
  void** allocations;
};

class BM_malloc : public ::allocatorBenchmark {
  public:
  virtual void* allocate( size_t nbytes ) { return malloc(nbytes); }
  virtual void deallocate( void* ptr ) { free(ptr); }
};
BENCHMARK_DEFINE_F(BM_malloc, malloc)(benchmark::State &st) { BM_allocation(st); }
BENCHMARK_DEFINE_F(BM_malloc, free)(benchmark::State &st)   { BM_deallocation(st); }

class BM_allocAlloc : public ::allocatorBenchmark {
  public:
  void SetUp(const ::benchmark::State&) {
    auto& rm = umpire::ResourceManager::getInstance();
    allocator = new umpire::Allocator(rm.getAllocator(getName()));
  }
  void TearDown(const ::benchmark::State&) {
    delete allocator;
  }
  virtual void* allocate( size_t nbytes ) { return allocator->allocate(nbytes); }
  virtual void deallocate( void* ptr ) { allocator->deallocate(ptr); }
  virtual const std::string& getName( void ) = 0;

  umpire::Allocator* allocator;
};

class BM_allocateHost : public ::BM_allocAlloc {
  public:
    BM_allocateHost(): name("HOST") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(BM_allocateHost, allocate)(benchmark::State &st) { BM_allocation(st); }
BENCHMARK_DEFINE_F(BM_allocateHost, deallocate)(benchmark::State &st)   { BM_deallocation(st); }

class BM_allocateDevice : public ::BM_allocAlloc {
  public:
    BM_allocateDevice(): name("DEVICE") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(BM_allocateDevice, allocate)(benchmark::State &st) { BM_allocation(st); }
BENCHMARK_DEFINE_F(BM_allocateDevice, deallocate)(benchmark::State &st)   { BM_deallocation(st); }

class BM_allocateUM : public ::BM_allocAlloc {
  public:
    BM_allocateUM(): name("UM") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(BM_allocateUM, allocate)(benchmark::State &st) { BM_allocation(st); }
BENCHMARK_DEFINE_F(BM_allocateUM, deallocate)(benchmark::State &st)   { BM_deallocation(st); }

static int namecnt = 0;   // Used to generate unique name per iteration
class BM_allocatePool : public ::allocatorBenchmark {
  public:
  void SetUp(const ::benchmark::State&) {
    std::stringstream ss;
    ss << "host_pool" << namecnt++;
    auto& rm = umpire::ResourceManager::getInstance();
    rm.makeAllocator<umpire::strategy::DynamicPool>(ss.str(), rm.getAllocator(getName()), (max_allocations+1)*1024);
    allocator = new umpire::Allocator(rm.getAllocator(ss.str()));

    void* ptr;
    ptr = allocate(100);
    deallocate(ptr);
  }

  void TearDown(const ::benchmark::State&) {
    delete allocator;
  }
  virtual void* allocate( size_t nbytes ) { return allocator->allocate(nbytes); }
  virtual void deallocate( void* ptr ) { allocator->deallocate(ptr); }
  virtual const std::string& getName( void ) = 0;

  umpire::Allocator* allocator;
};

class BM_allocatePoolHost : public ::BM_allocatePool {
  public:
    BM_allocatePoolHost(): name("HOST") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(BM_allocatePoolHost, allocate)(benchmark::State &st) { BM_allocation(st); }
BENCHMARK_DEFINE_F(BM_allocatePoolHost, deallocate)(benchmark::State &st)   { BM_deallocation(st); }

class BM_allocatePoolDevice : public ::BM_allocatePool {
  public:
    BM_allocatePoolDevice(): name("DEVICE") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(BM_allocatePoolDevice, allocate)(benchmark::State &st) { BM_allocation(st); }
BENCHMARK_DEFINE_F(BM_allocatePoolDevice, deallocate)(benchmark::State &st)   { BM_deallocation(st); }

class BM_allocatePoolUM : public ::BM_allocatePool {
  public:
    BM_allocatePoolUM(): name("UM") { }
    const std::string& getName( void ) { return name; }
  private:
    const std::string name;
};
BENCHMARK_DEFINE_F(BM_allocatePoolUM, allocate)(benchmark::State &st) { BM_allocation(st); }
BENCHMARK_DEFINE_F(BM_allocatePoolUM, deallocate)(benchmark::State &st)   { BM_deallocation(st); }

static const int RangeLow = 4;
static const int RangeHi = 1024;

BENCHMARK_REGISTER_F(BM_malloc, malloc)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_malloc, free)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocateHost, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocateHost, deallocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocatePoolHost, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocatePoolHost, deallocate)->Range(RangeLow, RangeHi);

#if defined(UMPIRE_ENABLE_CUDA)
BENCHMARK_REGISTER_F(BM_allocateDevice, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocateDevice, deallocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocatePoolDevice, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocatePoolDevice, deallocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocateUM, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocateUM, deallocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocatePoolUM, allocate)->Range(RangeLow, RangeHi);
BENCHMARK_REGISTER_F(BM_allocatePoolUM, deallocate)->Range(RangeLow, RangeHi);
#endif

BENCHMARK_MAIN()
