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
#include <vector>
#include <string>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

class replayTest {
public:
  replayTest() : , 
  {
    auto& rm = umpire::ResourceManager::getInstance();

    rm.makeAllocator<umpire::strategy::DynamicPool>(
        "host_simpool_defaults", rm.getAllocator("HOST"));
    allocatorNames.emplace_back("host_simpool_defaults");

    rm.makeAllocator<umpire::strategy::DynamicPool>(
        "host_simpool_spec1", rm.getAllocator("HOST"), 9876, 1234);
    allocatorNames.emplace_back("host_simpool_spec1");

    rm.makeAllocator<umpire::strategy::DynamicPool, false>(
        "host_simpool_spec2", rm.getAllocator("HOST"), 9876, 1234);
    allocatorNames.emplace_back("host_simpool_spec2");

    rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "MONOTONIC 1024", 1024, rm.getAllocator("HOST"));
    allocatorNames.emplace_back("MONOTONIC 1024");

    rm.makeAllocator<umpire::strategy::SlotPool>(
      "host_slot_pool", 64, rm.getAllocator("HOST"));
    allocatorNames.emplace_back("host_slot_pool");

    rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "thread_safe_allocator", rm.getAllocator("HOST"));
    allocatorNames.emplace_back("thread_safe_allocator");

#if 0
    //
    // Replay currently cannot support replaying FixedPool allocations.
    // This is because replay does its work at runtime and the FixedPool
    // is a template where sizes are generated at compile time.
    //
    struct data { char _[1024*1024]; };

    rm.makeAllocator<umpire::strategy::FixedPool<data>>(
        "fixed_pool_allocator", rm.getAllocator("HOST"));
    allocatorNames.push_back("fixed_pool_allocator");
#endif
  }

  ~replayTest( )
  = default;

  void runTest()
  {
    auto& rm = umpire::ResourceManager::getInstance();
    auto pooled_allocator = rm.getAllocator("host_simpool_spec1");
    auto strategy = pooled_allocator.getAllocationStrategy();
    auto tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(strategy);

    if (tracker) {
      strategy = tracker->getAllocationStrategy();
    }

    auto dynamic_pool = dynamic_cast<umpire::strategy::DynamicPool*>(strategy);

    if (! dynamic_pool ) {
      std::cerr << "host_simpool_spec1 is not a dynamic pool!\n";
      exit(1);
    }

    for ( int i = 0; i < testAllocations; ++i ) {
      for ( auto n : allocatorNames ) {
        auto alloc = rm.getAllocator(n);
        allocations.emplace_back(alloc.allocate( ++allocationSize ), n );
      }
    }

    dynamic_pool->coalesce();

    for ( auto ptr : allocations ) {
      auto alloc = rm.getAllocator(ptr.second);
      alloc.deallocate( ptr.first );
    }

    dynamic_pool->coalesce();
    pooled_allocator.release();
  }
private:
  const int testAllocations{3};
  std::size_t allocationSize{16};
  std::vector<std::string> allocatorNames;
  std::vector<std::pair<void*, std::string>> allocations;
};


int main(int , char** )
{
  replayTest test;

  test.runTest();

  return 0;
}
