//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"

#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"

void alloc_leak_example()
{
  std::cout << std::endl << std::endl<< "Allocation Leak Example: " << std::endl;
  auto& rm = umpire::ResourceManager::getInstance();

  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_dynamic_pool", rm.getAllocator("HOST"));

  auto alloc1 = pool.allocate(24);
  auto alloc2 = pool.allocate(64);
  auto alloc3 = pool.allocate(128);

  std::stringstream ss;
  umpire::print_allocator_records(pool, ss);

  if (! ss.str().empty() )
    std::cout << ss.str();

  pool.deallocate(alloc1);
  pool.deallocate(alloc2);
  pool.deallocate(alloc3);

  ss.str("");
  umpire::print_allocator_records(pool, ss);

  if (! ss.str().empty() )
    std::cout << ss.str() << std::endl;
}

void umpire_exception_example()
{
  std::cout << std::endl << std::endl<< "Umpire Exception Example: " << std::endl;
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");
  auto alloc1 = allocator.allocate(24);

  allocator.deallocate(alloc1);

  // Will throw exception
  //
  try {
    allocator.deallocate(alloc1);
  }
  catch (const std::exception &exc) {
    std::cout << "Exception thrown from Umpire:" << std::endl
      << exc.what();
  }
}

int main(int, char**)
{
  alloc_leak_example();
  umpire_exception_example();
  return 0;
}
