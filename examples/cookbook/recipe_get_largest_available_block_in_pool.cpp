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
#include "umpire/util/Exception.hpp"
#include "umpire/util/wrap_allocator.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  // _sphinx_tag_tut_unwrap_start
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>(
      "pool", rm.getAllocator("HOST"));

  auto quick_pool =
      umpire::util::unwrap_allocator<umpire::strategy::QuickPool>(pool);
  // _sphinx_tag_tut_unwrap_end

  if (quick_pool == nullptr) {
    UMPIRE_ERROR(pool.getName() << " is not a QuickPool");
  }

  auto ptr = pool.allocate(1024);

  // _sphinx_tag_tut_get_info_start
  std::cout << "Largest available block in pool is "
            << quick_pool->getLargestAvailableBlock() << " bytes in size"
            << std::endl;
  // _sphinx_tag_tut_get_info_end

  pool.deallocate(ptr);

  return 0;
}
