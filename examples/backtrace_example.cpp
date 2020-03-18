//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <sstream>

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"

#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"

#ifdef UMPIRE_ENABLE_ALLOCATION_BACKTRACE
void alloc_leak_example()
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_dynamic_pool", rm.getAllocator("HOST"));

  pool.allocate(24);
  pool.allocate(64);
  pool.allocate(128);

  std::stringstream ss;
  umpire::print_allocator_records(pool, ss);

  if (! ss.str().empty() )
    std::cout << ss.str();
}
#endif // UMPIRE_ENABLE_ALLOCATION_BACKTRACE

void umpire_exception_example()
{
  auto allocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
  auto allocation = allocator.allocate(24);

  allocator.deallocate(allocation);

  try {
    allocator.deallocate(allocation); // Will throw error
  }
  catch (const std::exception &exc) {
    std::cout << "Exception thrown from Umpire:" << std::endl << exc.what();
  }
}

int main(int, char**)
{
#ifdef UMPIRE_ENABLE_ALLOCATION_BACKTRACE
  alloc_leak_example();
#endif // UMPIRE_ENABLE_ALLOCATION_BACKTRACE
  umpire_exception_example();
  return 0;
}
