//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <vector>
#include <string>

#include "umpire/config.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/MixedPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/util/wrap_allocator.hpp"

class replayTest {
public:
  replayTest() : testAllocations(3), allocationSize(16)
  {
    auto& rm = umpire::ResourceManager::getInstance();

    allocatorNames.push_back("HOST");
#if defined(UMPIRE_ENABLE_DEVICE)
    allocatorNames.push_back("DEVICE");
#endif
#if defined(UMPIRE_ENABLE_UM)
    allocatorNames.push_back("UM");
#endif
#if defined(UMPIRE_ENABLE_PINNED)
    allocatorNames.push_back("PINNED");
#endif

    rm.makeAllocator<umpire::strategy::MixedPool>(
          "host_mixedpool_default"
        , rm.getAllocator("HOST")
        //, 512             // smallest fixed block size (Bytes)
        //, 1*1024          // largest fixed block size 1KiB
        //, 4 * 1024 * 1024 // max fixed pool size
        //, 12.0            // size multiplier
        //, (256 * 1024 * 1204) // dynamic pool min initial allocation size
        //, (1 * 1024 * 1024)   // dynamic pool min allocation size
        // , umpire::strategy::heuristic_percent_releasable(75)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_mixedpool_default");

    rm.makeAllocator<umpire::strategy::MixedPool>(
          "host_mixedpool_spec1"
        , rm.getAllocator("HOST")
        , 512             // smallest fixed block size (Bytes)
        //, 1*1024          // largest fixed block size 1KiB
        //, 4 * 1024 * 1024 // max fixed pool size
        //, 12.0            // size multiplier
        //, (256 * 1024 * 1204) // dynamic pool min initial allocation size
        //, (1 * 1024 * 1024)   // dynamic pool min allocation size
        // , umpire::strategy::heuristic_percent_releasable(75)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_mixedpool_spec1");

    rm.makeAllocator<umpire::strategy::MixedPool>(
          "host_mixedpool_spec2"
        , rm.getAllocator("HOST")
        , 512             // smallest fixed block size (Bytes)
        , 1*1024          // largest fixed block size 1KiB
        //, 4 * 1024 * 1024 // max fixed pool size
        //, 12.0            // size multiplier
        //, (256 * 1024 * 1204) // dynamic pool min initial allocation size
        //, (1 * 1024 * 1024)   // dynamic pool min allocation size
        // , umpire::strategy::heuristic_percent_releasable(75)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_mixedpool_spec2");

    rm.makeAllocator<umpire::strategy::MixedPool>(
          "host_mixedpool_spec3"
        , rm.getAllocator("HOST")
        , 512             // smallest fixed block size (Bytes)
        , 1*1024          // largest fixed block size 1KiB
        , 4 * 1024 * 1024 // max fixed pool size
        //, 12.0            // size multiplier
        //, (256 * 1024 * 1204) // dynamic pool min initial allocation size
        //, (1 * 1024 * 1024)   // dynamic pool min allocation size
        // , umpire::strategy::heuristic_percent_releasable(75)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_mixedpool_spec3");

    rm.makeAllocator<umpire::strategy::MixedPool>(
          "host_mixedpool_spec4"
        , rm.getAllocator("HOST")
        , 512             // smallest fixed block size (Bytes)
        , 1*1024          // largest fixed block size 1KiB
        , 4 * 1024 * 1024 // max fixed pool size
        , 12              // size multiplier
        //, 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        // , umpire::strategy::heuristic_percent_releasable(75)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_mixedpool_spec4");

    rm.makeAllocator<umpire::strategy::MixedPool>(
          "host_mixedpool_spec5"
        , rm.getAllocator("HOST")
        , 512             // smallest fixed block size (Bytes)
        , 1*1024          // largest fixed block size 1KiB
        , 4 * 1024 * 1024 // max fixed pool size
        , 12              // size multiplier
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        // , umpire::strategy::heuristic_percent_releasable(75)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_mixedpool_spec5");

    rm.makeAllocator<umpire::strategy::MixedPool>(
          "host_mixedpool_spec6"
        , rm.getAllocator("HOST")
        , 512             // smallest fixed block size (Bytes)
        , 1*1024          // largest fixed block size 1KiB
        , 4 * 1024 * 1024 // max fixed pool size
        , 12              // size multiplier
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        // , umpire::strategy::heuristic_percent_releasable(75)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_mixedpool_spec6");

    rm.makeAllocator<umpire::strategy::MixedPool>(
          "host_mixedpool_spec7"
        , rm.getAllocator("HOST")
        , 512             // smallest fixed block size (Bytes)
        , 1*1024          // largest fixed block size 1KiB
        , 4 * 1024 * 1024 // max fixed pool size
        , 12              // size multiplier
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(75)
    );
    allocatorNames.push_back("host_mixedpool_spec7");

    rm.makeAllocator<umpire::strategy::MixedPool>(
          "host_mixedpool_spec8"
        , rm.getAllocator("HOST")
        , 512             // smallest fixed block size (Bytes)
        , 1*1024          // largest fixed block size 1KiB
        , 4 * 1024 * 1024 // max fixed pool size
        , 12              // size multiplier
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        , umpire::strategy::heuristic_percent_releasable(75)
    );
    allocatorNames.push_back("host_mixedpool_spec8");

    //
    // DynamicPool
    //
    rm.makeAllocator<umpire::strategy::DynamicPool>(
          "host_dyn_pool_spec0", rm.getAllocator("HOST")
        //, 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, umpire::strategy::heuristic_percent_releasable(50)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_dyn_pool_spec0");

    rm.makeAllocator<umpire::strategy::DynamicPool>(
          "host_dyn_pool_spec1", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, umpire::strategy::heuristic_percent_releasable(50)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_dyn_pool_spec1");

    rm.makeAllocator<umpire::strategy::DynamicPool>(
          "host_dyn_pool_spec2", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        //, umpire::strategy::heuristic_percent_releasable(50)
        //, 128                 // byte alignment
    );
    allocatorNames.push_back("host_dyn_pool_spec2");

    rm.makeAllocator<umpire::strategy::DynamicPool>(
          "host_dyn_pool_spec3", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_spec3");

    rm.makeAllocator<umpire::strategy::DynamicPool>(
          "host_dyn_pool_spec4", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        , umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_spec4");

    rm.makeAllocator<umpire::strategy::DynamicPool, false>(
          "host_dyn_pool_nointro_spec0", rm.getAllocator("HOST")
        //, 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_nointro_spec0");

    rm.makeAllocator<umpire::strategy::DynamicPool, false>(
          "host_dyn_pool_nointro_spec1", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_nointro_spec1");

    rm.makeAllocator<umpire::strategy::DynamicPool, false>(
          "host_dyn_pool_nointro_spec2", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        //, 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_nointro_spec2");

    rm.makeAllocator<umpire::strategy::DynamicPool, false>(
          "host_dyn_pool_nointro_spec3", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_nointro_spec3");

    rm.makeAllocator<umpire::strategy::DynamicPool, false>(
          "host_dyn_pool_nointro_spec4", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        , umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_nointro_spec4");


    //
    // DynamicPoolMap
    //
    rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
          "host_dyn_pool_map_spec0", rm.getAllocator("HOST")
        //, 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_spec0");

    rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
          "host_dyn_pool_map_spec1", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_spec1");

    rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
          "host_dyn_pool_map_spec2", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        //, 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_spec2");

    rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
          "host_dyn_pool_map_spec3", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_spec3");

    rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
          "host_dyn_pool_map_spec4", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        , umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_spec4");

    rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>(
          "host_dyn_pool_map_nointro_spec0", rm.getAllocator("HOST")
        //, 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_nointro_spec0");

    rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>(
          "host_dyn_pool_map_nointro_spec1", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_nointro_spec1");

    rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>(
          "host_dyn_pool_map_nointro_spec2", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        //, 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_nointro_spec2");

    rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>(
          "host_dyn_pool_map_nointro_spec3", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        //, umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_nointro_spec3");

    rm.makeAllocator<umpire::strategy::DynamicPoolMap, false>(
          "host_dyn_pool_map_nointro_spec4", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , 128                 // byte alignment
        , umpire::strategy::heuristic_percent_releasable(50)
    );
    allocatorNames.push_back("host_dyn_pool_map_nointro_spec4");

    //
    // DynamicPoolList
    //
    rm.makeAllocator<umpire::strategy::DynamicPoolList>(
          "host_dyn_pool_list_spec0", rm.getAllocator("HOST")
        //, 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, umpire::strategy::heuristic_percent_releasable_list(50)
    );
    allocatorNames.push_back("host_dyn_pool_list_spec0");

    rm.makeAllocator<umpire::strategy::DynamicPoolList>(
          "host_dyn_pool_list_spec1", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, umpire::strategy::heuristic_percent_releasable_list(50)
    );
    allocatorNames.push_back("host_dyn_pool_list_spec1");

    rm.makeAllocator<umpire::strategy::DynamicPoolList>(
          "host_dyn_pool_list_spec2", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        //, umpire::strategy::heuristic_percent_releasable_list(50)
    );
    allocatorNames.push_back("host_dyn_pool_list_spec2");

    rm.makeAllocator<umpire::strategy::DynamicPoolList>(
          "host_dyn_pool_list_spec3", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , umpire::strategy::heuristic_percent_releasable_list(50)
    );
    allocatorNames.push_back("host_dyn_pool_list_spec3");

    rm.makeAllocator<umpire::strategy::DynamicPoolList, false>(
          "host_dyn_pool_list_nointro_spec0", rm.getAllocator("HOST")
        //, 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, umpire::strategy::heuristic_percent_releasable_list(50)
    );
    allocatorNames.push_back("host_dyn_pool_list_nointro_spec0");

    rm.makeAllocator<umpire::strategy::DynamicPoolList, false>(
          "host_dyn_pool_list_nointro_spec1", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        //, 1 * 1024 * 1024   // dynamic pool min allocation size
        //, umpire::strategy::heuristic_percent_releasable_list(50)
    );
    allocatorNames.push_back("host_dyn_pool_list_nointro_spec1");

    rm.makeAllocator<umpire::strategy::DynamicPoolList, false>(
          "host_dyn_pool_list_nointro_spec2", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        //, umpire::strategy::heuristic_percent_releasable_list(50)
    );
    allocatorNames.push_back("host_dyn_pool_list_nointro_spec2");

    rm.makeAllocator<umpire::strategy::DynamicPoolList, false>(
          "host_dyn_pool_list_nointro_spec3", rm.getAllocator("HOST")
        , 256 * 1024 * 1024 // dynamic pool min initial allocation size
        , 1 * 1024 * 1024   // dynamic pool min allocation size
        , umpire::strategy::heuristic_percent_releasable_list(50)
    );
    allocatorNames.push_back("host_dyn_pool_list_nointro_spec3");

    //
    // Monotonic Allocation
    //
    rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "MONOTONIC 1024", rm.getAllocator("HOST"), 1024);
    allocatorNames.push_back("MONOTONIC 1024");

    //
    // Slot
    //
    rm.makeAllocator<umpire::strategy::SlotPool>(
      "host_slot_pool", rm.getAllocator("HOST"), 64);
    allocatorNames.push_back("host_slot_pool");

    rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "thread_safe_allocator", rm.getAllocator("HOST"));
    allocatorNames.push_back("thread_safe_allocator");

    //
    // Fixed
    //
    rm.makeAllocator<umpire::strategy::FixedPool>(
          "host_fixed_pool_spec0", rm.getAllocator("HOST")
        , 32                // object_bytes
        //, 1024              // objects_per_pool                        
    );
    FixedAllocatorNames.push_back("host_fixed_pool_spec0");

    rm.makeAllocator<umpire::strategy::FixedPool>(
          "host_fixed_pool_spec1", rm.getAllocator("HOST")
        , 32                // object_bytes
        , 1024              // objects_per_pool                        
    );
    FixedAllocatorNames.push_back("host_fixed_pool_spec1");

    rm.makeAllocator<umpire::strategy::FixedPool, false>(
          "host_fixed_pool_nointro_spec1", rm.getAllocator("HOST")
        , 32                // object_bytes
        , 1024              // objects_per_pool                        
    );
    FixedAllocatorNames.push_back("host_fixed_pool_nointro_spec1");
  }

  ~replayTest( void )
  {
  }

  void runTest()
  {
    auto& rm = umpire::ResourceManager::getInstance();

    auto pooled_allocator = rm.getAllocator("host_dyn_pool_spec1");
    auto dynamic_pool = 
      umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(pooled_allocator);

    if (! dynamic_pool ) {
      std::cerr << "host_dyn_pool_spec1 is not a dynamic pool!\n";
      exit(1);
    }

    for ( int i = 0; i < testAllocations; ++i ) {
      for ( auto n : allocatorNames ) {
        auto alloc = rm.getAllocator(n);
        allocations.push_back( std::make_pair(alloc.allocate( ++allocationSize ), n) );
      }
    }

    for ( int i = 0; i < testAllocations; ++i ) {
      for ( auto n : FixedAllocatorNames ) {
        auto alloc = rm.getAllocator(n);
        allocations.push_back( std::make_pair(alloc.allocate(32), n) );
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
  const int testAllocations;
  std::size_t allocationSize;
  std::vector<std::string> allocatorNames;
  std::vector<std::string> FixedAllocatorNames;
  std::vector<std::pair<void*, std::string>> allocations;
};


int main(int , char** )
{
  replayTest test;

  test.runTest();

  return 0;
}
