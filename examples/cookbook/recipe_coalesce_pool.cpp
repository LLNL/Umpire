//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AllocationTracker.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/util/Exception.hpp"
#include "umpire/util/wrap_allocator.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "pool", rm.getAllocator("HOST"));

  // _sphinx_tag_tut_unwrap_strategy_start
  auto dynamic_pool =
      umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(pool);
  // _sphinx_tag_tut_unwrap_strategy_end

  if (dynamic_pool) {
    // _sphinx_tag_tut_call_coalesce_start
    dynamic_pool->coalesce();
    // _sphinx_tag_tut_call_coalesce_end
  } else {
    UMPIRE_ERROR(pool.getName() << " is not a DynamicPool, cannot coalesce!");
  }

  return 0;
}
