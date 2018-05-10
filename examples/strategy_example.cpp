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

#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/Pool.hpp"

#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/GenericAllocationStrategyFactory.hpp"
#include "umpire/strategy/AllocationStrategyRegistry.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  std::cout << "Available allocators: ";
  for (auto s : rm.getAvailableAllocators()){
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  /*
   * Register AllocationStrategies.
   *
   * For Umpire-provided AllocationStrategies this step will be hidden. For
   * user-defined strategies, a Factory must be registered. A Generic factory
   * is provided for simple strategies.
   *
   */
  auto& alloc_registry = umpire::strategy::AllocationStrategyRegistry::getInstance();

  alloc_registry.registerAllocationStrategy(
      std::make_shared<umpire::strategy::GenericAllocationStrategyFactory<umpire::strategy::Pool> >("POOL"));

  alloc_registry.registerAllocationStrategy(
      std::make_shared<
        umpire::strategy::GenericAllocationStrategyFactory<
          umpire::strategy::MonotonicAllocationStrategy> >("MONOTONIC"));

  /*
   * Build some new Allocators from the named AllocationStrategies.
   *
   * Allocator makeAllocator(
   *     const std::string& name,  // Allocator name
   *     const std::string& strategy,  // Strategy name
   *     AllocatorTraits traits, // Traits object
   *     std::vector<Allocator> providers); // Vector of providers (parent) Allocators
   *
   *  Named Allocators are stored in a map, and can be later accessed using the
   *  getAllocator function.
   */
  auto alloc = rm.makeAllocator("POOL", "POOL", {0,0,64}, {rm.getAllocator("HOST")});
  alloc = rm.makeAllocator("MONOTONIC 1024", "MONOTONIC", {1024,0,0}, {rm.getAllocator("HOST")});
  alloc = rm.makeAllocator("MONOTONIC 4096", "MONOTONIC", {4096,0,0}, {rm.getAllocator("HOST")});



  /*
   * Get the previously created POOL allocator.
   *
   * Allocator getAllocator(const std::string& name); // name of allocator
   *
   */
  alloc = rm.getAllocator("POOL");
  void* test = alloc.allocate(100);
  alloc.deallocate(test);


  /*
   * Get the default HOST allocator.
   */
  alloc = rm.getAllocator("HOST");
  test = alloc.allocate(100);
  alloc.deallocate(test);

  /*
   * Get the previously created MONOTONIC allocator..
   */
  alloc = rm.getAllocator("MONOTONIC 1024");
  test = alloc.allocate(14);
  alloc.deallocate(test);

  std::cout << "Size: " << alloc.getSize(test) << std::endl;

  std::cout << "Available allocators: ";
  for (auto s : rm.getAvailableAllocators()){
    std::cout << s << ", ";
  }
  std::cout << std::endl;

  return 0;
}
