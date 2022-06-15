//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
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

  pool_allocator.allocate(128 + 64);
  pool_allocator.allocate(256 + 32);
  pool_allocator.allocate(512 + 16);

  allocator.deallocate((void*)32);
  allocator.allocate(128);

  pool_allocator.deallocate((void*)512);
  pool_allocator.allocate(256);

  std::stringstream ss;
  umpire::print_allocator_records(allocator, ss);
  umpire::print_allocator_records(pool_allocator, ss);

  if (!ss.str().empty())
    std::cout << ss.str();

  // When umpire throws an exception, a backtrace to the offending call will
  // be provided in the exception string.
  void* bad_ptr = (void*)0xBADBADBAD;

  try {
    allocator.deallocate(bad_ptr); // Will cause a throw from umpire
  } catch (const std::exception& exc) {
    std::cout << "Exception thrown from Umpire:" << std::endl << exc.what();
  }

  return 0;
}
