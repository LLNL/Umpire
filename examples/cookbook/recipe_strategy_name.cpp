//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  // _sphinx_tag_tut_strategy_name_start
  //
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>(
      "POOL", allocator);
  std::cout << pool.getStrategyName() << std::endl;
  // _sphinx_tag_tut_strategy_name_end

  UMPIRE_USE_VAR(pool);

  return 0;
}
