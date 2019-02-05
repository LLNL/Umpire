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
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

#include <iostream>

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("UM");

  /*
   * Create an allocator that applied "PREFFERED_LOCATION" advice to set the
   * GPU as the preferred location.
   */
  auto preferred_location_allocator =
    rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
      "preferred_location_host", allocator, "PREFERRED_LOCATION");

  /* 
   * Create a pool using the preferred_location_allocator. This makes all
   * allocations in the pool have the same preferred location, the GPU.
   */
  auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
                            "GPU_POOL",
                            preferred_location_allocator);

  return 0;
}
