//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/wrap_allocator.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool", rm.getAllocator("HOST"));

  // _sphinx_tag_tut_unwrap_strategy_start
  auto quick_pool = umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool);
  // _sphinx_tag_tut_unwrap_strategy_end

  if (quick_pool) {
    // _sphinx_tag_tut_call_coalesce_start
    quick_pool->coalesce();
    // _sphinx_tag_tut_call_coalesce_end
  } else {
    UMPIRE_ERROR(umpire::runtime_error, fmt::format("{} is not a QuickPool, cannot coalesce!", pool.getName()));
  }

  return 0;
}
