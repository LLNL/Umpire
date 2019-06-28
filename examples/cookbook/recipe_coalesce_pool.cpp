//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/AllocationTracker.hpp"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/util/Exception.hpp"

#include <iostream>

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();

  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "pool", rm.getAllocator("HOST"));

  auto strategy = pool.getAllocationStrategy();
  auto tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(strategy);

  if (tracker) {
    strategy = tracker->getAllocationStrategy();
  }

  auto dynamic_pool = dynamic_cast<umpire::strategy::DynamicPool*>(strategy);

  if (dynamic_pool) {
    dynamic_pool->coalesce();
  } else {
    UMPIRE_ERROR(pool.getName() << " is not a DynamicPool, cannot coalesce!");
  }

  return 0;
}

