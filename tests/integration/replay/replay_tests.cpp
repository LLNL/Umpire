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
  replayTest() : test_allocations(3), allocation_size(32)
  {
    auto& rm = umpire::ResourceManager::getInstance();

    std::vector<std::string> allocators{
        "HOST" 
#if defined(UMPIRE_ENABLE_DEVICE)
      , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
      , "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
      , "PINNED"
#endif
    };

    for ( auto basename : allocators ) {
      std::string name;
      allocator_names.push_back(basename);
      auto base_alloc = rm.getAllocator(basename);

#if defined(UMPIRE_ENABLE_CUDA)
      if ( basename == "UM" ) {
        auto device_id = 1;   // device_id
        auto accessing_alloc = rm.getAllocator("HOST");

        name = basename + "_AllocationAdvisor_spec_";
        makeAllocator<umpire::strategy::AllocationAdvisor, true>(name+"default_id", base_alloc, "READ_MOSTLY");
        makeAllocator<umpire::strategy::AllocationAdvisor, true>(name+"with_id", base_alloc, "READ_MOSTLY", device_id);
        makeAllocator<umpire::strategy::AllocationAdvisor, true>(name+"with_accessing_and_default_id", base_alloc, "READ_MOSTLY", accessing_alloc);
        makeAllocator<umpire::strategy::AllocationAdvisor, true>(name+"with_accessing_and_id", base_alloc, "PREFERRED_LOCATION", accessing_alloc, device_id);
        name = basename + "_AllocationAdvisor_no_introspection_spec_";
        makeAllocator<umpire::strategy::AllocationAdvisor, false>(name+"default_id", base_alloc, "READ_MOSTLY");
        makeAllocator<umpire::strategy::AllocationAdvisor, false>(name+"with_id", base_alloc, "READ_MOSTLY", device_id);
        makeAllocator<umpire::strategy::AllocationAdvisor, false>(name+"with_accessing_and_default_id", base_alloc, "READ_MOSTLY", accessing_alloc);
        makeAllocator<umpire::strategy::AllocationAdvisor, false>(name+"with_accessing_and_id", base_alloc, "PREFERRED_LOCATION", accessing_alloc, device_id);
    }
#endif // defined(UMPIRE_ENABLE_CUDA)

      auto mpa1 = 512;                // Smallest fixed pool object size in bytes
      auto mpa2 = 1*1024;             // Largest fixed pool object size in bytes
      auto mpa3 = 4 * 1024 * 1024;    // Largest initial size of any fixed pool
      auto mpa4 = 12;                 // Fixed pool object size increase factor
      auto mpa5 = 256 * 1024 * 1024;  // Size the dynamic pool initially allocates
      auto mpa6 = 1 * 1024 * 1024;    // Minimum size of all future allocations in the dynamic pool
      auto mpa7 = 128;                // Size with which to align allocations
      auto mpa8 = umpire::strategy::heuristic_percent_releasable(75); // Heuristic
      name = basename + "_MixedPool_spec_";
      makeAllocator<umpire::strategy::MixedPool, true>(name+"0", base_alloc);
      makeAllocator<umpire::strategy::MixedPool, true>(name+"1", base_alloc , mpa1);
      makeAllocator<umpire::strategy::MixedPool, true>(name+"2", base_alloc, mpa1, mpa2);
      makeAllocator<umpire::strategy::MixedPool, true>(name+"3", base_alloc, mpa1, mpa2, mpa3);
      makeAllocator<umpire::strategy::MixedPool, true>(name+"4", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5);
      makeAllocator<umpire::strategy::MixedPool, true>(name+"5", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5);
      makeAllocator<umpire::strategy::MixedPool, true>(name+"6", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5, mpa6);
      makeAllocator<umpire::strategy::MixedPool, true>(name+"7", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5, mpa6, mpa7);
      makeAllocator<umpire::strategy::MixedPool, true>(name+"8", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5, mpa6, mpa7, mpa8);
      name = basename + "_MixedPool_no_instrospection_spec_";
      makeAllocator<umpire::strategy::MixedPool, false>(name+"0", base_alloc);
      makeAllocator<umpire::strategy::MixedPool, false>(name+"1", base_alloc , mpa1);
      makeAllocator<umpire::strategy::MixedPool, false>(name+"2", base_alloc, mpa1, mpa2);
      makeAllocator<umpire::strategy::MixedPool, false>(name+"3", base_alloc, mpa1, mpa2, mpa3);
      makeAllocator<umpire::strategy::MixedPool, false>(name+"4", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5);
      makeAllocator<umpire::strategy::MixedPool, false>(name+"5", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5);
      makeAllocator<umpire::strategy::MixedPool, false>(name+"6", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5, mpa6);
      makeAllocator<umpire::strategy::MixedPool, false>(name+"7", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5, mpa6, mpa7);
      makeAllocator<umpire::strategy::MixedPool, false>(name+"8", base_alloc, mpa1, mpa2, mpa3, mpa4, mpa5, mpa6, mpa7, mpa8);

      auto dpa1 = 256 * 1024 * 1024; // min initial allocation size
      auto dpa2 = 1 * 1024 * 1024;   // min allocation size
      auto dpa3 = 128;               // byte alignment
      auto dpa4 = umpire::strategy::heuristic_percent_releasable(50);
      name = basename + "_DynamicPool_spec_";
      makeAllocator<umpire::strategy::DynamicPool, true>(name+"0", base_alloc);
      makeAllocator<umpire::strategy::DynamicPool, true>(name+"1", base_alloc, dpa1);
      makeAllocator<umpire::strategy::DynamicPool, true>(name+"2", base_alloc, dpa1, dpa2);
      makeAllocator<umpire::strategy::DynamicPool, true>(name+"3", base_alloc, dpa1, dpa2, dpa3);
      makeAllocator<umpire::strategy::DynamicPool, true>(name+"4", base_alloc, dpa1, dpa2, dpa3, dpa4);
      name = basename + "_DynamicPool_no_instrospection_spec_";
      makeAllocator<umpire::strategy::DynamicPool, false>(name+"0", base_alloc);
      makeAllocator<umpire::strategy::DynamicPool, false>(name+"1", base_alloc, dpa1);
      makeAllocator<umpire::strategy::DynamicPool, false>(name+"2", base_alloc, dpa1, dpa2);
      makeAllocator<umpire::strategy::DynamicPool, false>(name+"3", base_alloc, dpa1, dpa2, dpa3);
      makeAllocator<umpire::strategy::DynamicPool, false>(name+"4", base_alloc, dpa1, dpa2, dpa3, dpa4);

      name = basename + "_DynamicPoolMap_spec_";
      makeAllocator<umpire::strategy::DynamicPoolMap, true>(name+"0", base_alloc);
      makeAllocator<umpire::strategy::DynamicPoolMap, true>(name+"1", base_alloc, dpa1);
      makeAllocator<umpire::strategy::DynamicPoolMap, true>(name+"2", base_alloc, dpa1, dpa2);
      makeAllocator<umpire::strategy::DynamicPoolMap, true>(name+"3", base_alloc, dpa1, dpa2, dpa3);
      makeAllocator<umpire::strategy::DynamicPoolMap, true>(name+"4", base_alloc, dpa1, dpa2, dpa3, dpa4);
      name = basename + "_DynamicPoolMap_no_instrospection_spec_";
      makeAllocator<umpire::strategy::DynamicPoolMap, false>(name+"0", base_alloc);
      makeAllocator<umpire::strategy::DynamicPoolMap, false>(name+"1", base_alloc, dpa1);
      makeAllocator<umpire::strategy::DynamicPoolMap, false>(name+"2", base_alloc, dpa1, dpa2);
      makeAllocator<umpire::strategy::DynamicPoolMap, false>(name+"3", base_alloc, dpa1, dpa2, dpa3);
      makeAllocator<umpire::strategy::DynamicPoolMap, false>(name+"4", base_alloc, dpa1, dpa2, dpa3, dpa4);

      auto lpa1 = dpa1;
      auto lpa2 = dpa2;
      auto lpa3 = umpire::strategy::heuristic_percent_releasable_list(50);
      name = basename + "_DynamicPoolList_spec_";
      makeAllocator<umpire::strategy::DynamicPoolList, true>(name+"0", base_alloc);
      makeAllocator<umpire::strategy::DynamicPoolList, true>(name+"1", base_alloc, lpa1);
      makeAllocator<umpire::strategy::DynamicPoolList, true>(name+"2", base_alloc, lpa1, lpa2);
      makeAllocator<umpire::strategy::DynamicPoolList, true>(name+"3", base_alloc, lpa1, lpa2, lpa3);
      name = basename + "_DynamicPoolList_no_instrospection_spec_";
      makeAllocator<umpire::strategy::DynamicPoolList, false>(name+"0", base_alloc);
      makeAllocator<umpire::strategy::DynamicPoolList, false>(name+"1", base_alloc, lpa1);
      makeAllocator<umpire::strategy::DynamicPoolList, false>(name+"2", base_alloc, lpa1, lpa2);
      makeAllocator<umpire::strategy::DynamicPoolList, false>(name+"3", base_alloc, lpa1, lpa2, lpa3);

      auto ma1 = 1024; // Capacity
      name = basename + "_MonotonicAllocationStrategy_spec_";
      makeAllocator<umpire::strategy::MonotonicAllocationStrategy, true>(name, base_alloc, ma1);
      name = basename + "_MonotonicAllocationStrategy_no_instrospection_spec_";
      makeAllocator<umpire::strategy::MonotonicAllocationStrategy, false>(name, base_alloc, ma1);

      auto sa1 = 64;  // Slots
      name = basename + "_SlotPool_spec_";
      makeAllocator<umpire::strategy::SlotPool, true>(name, base_alloc, sa1);
      name = basename + "_SlotPool_no_instrospection_spec_";
      makeAllocator<umpire::strategy::SlotPool, false>(name, base_alloc, sa1);

      name = basename + "_ThreadSafeAllocator_spec_";
      rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, true>(name, base_alloc);
      name = basename + "_ThreadSafeAllocator_no_instrospection_spec_";
      rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, false>(name, base_alloc);

      auto fpa1 = allocation_size; // object_bytes
      auto fpa2 = 1024; // objects_per_pool                        
      name = basename + "_FixedPool_spec_";
      makeAllocator<umpire::strategy::FixedPool, true>(name+"1", base_alloc, fpa1);
      makeAllocator<umpire::strategy::FixedPool, true>(name+"2", base_alloc, fpa1, fpa2);
      name = basename + "_FixedPool_no_instrospection_spec_";
      makeAllocator<umpire::strategy::FixedPool, false>(name+"1", base_alloc, fpa1);
      makeAllocator<umpire::strategy::FixedPool, false>(name+"2", base_alloc, fpa1, fpa2);
    }
  }

  ~replayTest( void )
  {
  }

  void runTest()
  {
    auto& rm = umpire::ResourceManager::getInstance();

    for ( int i = 0; i < test_allocations; ++i ) {
      for ( auto n : allocator_names ) {
        auto alloc = rm.getAllocator(n);
        allocations.push_back( std::make_pair(alloc.allocate( allocation_size ), n) );
      }
    }

    for ( auto ptr : allocations ) {
      auto alloc = rm.getAllocator(ptr.second);
      alloc.deallocate( ptr.first );
    }

    for ( auto n : allocator_names ) {
      try {
        auto alloc = rm.getAllocator(n);
        auto dynamic_pool = umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(alloc);

        dynamic_pool->coalesce();
        alloc.release();
      }
      catch ( ... ) {
      }
    }
  }

private:
  const int test_allocations;
  const std::size_t allocation_size;
  std::vector<std::string> allocator_names;
  std::vector<std::pair<void*, std::string>> allocations;

  template <typename Strategy,
           bool intro,
           typename... Args>
  void makeAllocator(std::string name, Args&&... args)
  {
    auto& rm = umpire::ResourceManager::getInstance();
    rm.makeAllocator<Strategy, intro>(name, args...);
    allocator_names.push_back(name);
  }
};


int main(int , char** )
{
  replayTest test;

  test.runTest();

  return 0;
}
