//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <sstream>

#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");
  auto pool_allocator = rm.makeAllocator<umpire::strategy::QuickPool>("host_quick_pool", allocator);

  allocator.allocate(16);
  allocator.allocate(32);
  allocator.allocate(64);

  pool_allocator.allocate(128);
  pool_allocator.allocate(256);
  pool_allocator.allocate(512);

  std::stringstream ss;
  umpire::print_allocator_records(allocator, ss);
  umpire::print_allocator_records(pool_allocator, ss);

  // Example #1 of 3 - Leaked allocations
  //
  // If Umpire compiled with -DUMPIRE_ENABLE_BACKTRACE=On, then backtrace
  // information will be printed for each of the allocations made above.
  //
  // Otherwise, if Umpire was not compiled with -DUMPIRE_ENABLE_BACKTRACE=On,
  // then only the addresses and size information for each allocation will be
  // printed.
  //
  if (!ss.str().empty())
    std::cout << ss.str();

  // Example #2 of 3 - Umpire error exceptions
  //
  // When umpire throws an exception, a backtrace to the offending call will
  // be provided in the exception string.
  //
  void* bad_ptr = (void*)0xBADBADBAD;

  try {
    allocator.deallocate(bad_ptr); // Will cause a throw from umpire
  } catch (const std::exception& exc) {
    //
    // exc.what() string will also contain a backtrace
    //
    std::cout << "Exception thrown from Umpire:" << std::endl << exc.what();
  }

  // Example #3 of 3 - Leak detection
  //
  // When the program terminates, Umpire's resource manager will be
  // deconstructed.  During deconstruction, Umpire will log the size and
  // address, of each leaked allocation in each allocator.
  //
  // If Umpire was compiled with -DUMPIRE_ENABLE_BACKTRACE=On, backtrace
  // information will also be logged for each leaked allocation in each
  // allocator.
  //
  // To enable (and see) the umpire logs, set the environment variable
  // UMPIRE_LOG_LEVEL=Error.
  //
  return 0;
}
