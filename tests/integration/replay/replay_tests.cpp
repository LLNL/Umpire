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

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/strategy/FixedPool.hpp"

class replayTest {
public:
  replayTest() : testAllocations(3), allocationSize(16)
  {
    auto& rm = umpire::ResourceManager::getInstance();

    rm.makeAllocator<umpire::strategy::DynamicPool>(
        "host_simpool_defaults", rm.getAllocator("HOST"));
    allocatorNames.push_back("host_simpool_defaults");

    rm.makeAllocator<umpire::strategy::DynamicPool>(
        "host_simpool_spec1", rm.getAllocator("HOST"), 9876, 1234);
    allocatorNames.push_back("host_simpool_spec1");

    rm.makeAllocator<umpire::strategy::DynamicPool, false>(
        "host_simpool_spec2", rm.getAllocator("HOST"), 9876, 1234);
    allocatorNames.push_back("host_simpool_spec2");

#if defined(UMPIRE_ENABLE_DEVICE)
    rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
      "read_only_um", rm.getAllocator("UM"), "READ_MOSTLY");
    allocatorNames.push_back("read_only_um");
#endif

#if defined(UMPIRE_ENABLE_UM)
    rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
      "preferred_location_host", rm.getAllocator("UM"),
      "PREFERRED_LOCATION", rm.getAllocator("HOST"));
    allocatorNames.push_back("preferred_location_host");
#endif

    rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "MONOTONIC 1024", 1024, rm.getAllocator("HOST"));
    allocatorNames.push_back("MONOTONIC 1024");

    rm.makeAllocator<umpire::strategy::SlotPool>(
      "host_slot_pool", 64, rm.getAllocator("HOST"));
    allocatorNames.push_back("host_slot_pool");

    rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "thread_safe_allocator", rm.getAllocator("HOST"));
    allocatorNames.push_back("thread_safe_allocator");

    struct data { char _[1024*1024]; };

    rm.makeAllocator<umpire::strategy::FixedPool<data>>(
        "fixed_pool_allocator", rm.getAllocator("HOST"));
    allocatorNames.push_back("fixed_pool_allocator");
  }

  ~replayTest( void )
  {
  }

  void runTest()
  {
    for ( int i = 0; i < testAllocations; ++i ) {
      for ( auto n : allocatorNames ) {
        auto& rm = umpire::ResourceManager::getInstance();
        auto alloc = rm.getAllocator(n);
        allocations.push_back( std::make_pair(alloc.allocate( ++allocationSize ), n) );
      }
    }

    for ( auto ptr : allocations ) {
      auto& rm = umpire::ResourceManager::getInstance();
      auto alloc = rm.getAllocator(ptr.second);
      alloc.deallocate( ptr.first );
    }
  }
private:
  const int testAllocations;
  std::size_t allocationSize;
  std::vector<std::string> allocatorNames;
  std::vector<std::pair<void*, std::string>> allocations;
};


int main(int , char** )
{
  replayTest test;

  test.runTest();

  return 0;
}
