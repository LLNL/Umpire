//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/util/Exception.hpp"
#include "umpire/util/wrap_allocator.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "pool", rm.getAllocator("HOST"));

  auto dynamic_pool =
      umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(pool);

  if (dynamic_pool == nullptr) {
    UMPIRE_ERROR(pool.getName() << " is not a DynamicPool");
  }

  auto ptr = pool.allocate(1024);

  std::cout << "Largest available block in pool is "
            << dynamic_pool->getLargestAvailableBlock() << " bytes in size"
            << std::endl;

  pool.deallocate(ptr);

  return 0;
}
