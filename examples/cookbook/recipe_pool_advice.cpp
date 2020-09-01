//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/util/Macros.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("UM");

  /*
   * Create an allocator that applied "PREFFERED_LOCATION" advice to set the
   * GPU as the preferred location.
   */
  auto preferred_location_allocator =
      rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
          "preferred_location_device", allocator, "PREFERRED_LOCATION");

  /*
   * Create a pool using the preferred_location_allocator. This makes all
   * allocations in the pool have the same preferred location, the GPU.
   */
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "GPU_POOL", preferred_location_allocator);

  UMPIRE_USE_VAR(pooled_allocator);

  return 0;
}
