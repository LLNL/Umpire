//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  // _sphinx_tag_tut_unwrap_strategy_start
  void* ptr{ allocator.allocate("My Allocation Name", 100) };

  auto record = rm.findAllocationRecord(ptr);
  std::cout << "The name of my allocation is: " << *(record->name) << std::endl;

  std::stringstream ss;
  umpire::print_allocator_records(allocator, ss);
  std::cout << "Tracked allocators are: " << std::endl << ss.str() << std::endl;

  allocator.deallocate(ptr);
  return 0;
}
