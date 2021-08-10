//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");
  std::vector<void*> allocations;

  allocations.push_back(allocator.allocate("My Allocation Name", 100));
  allocations.push_back(allocator.allocate(1024));

  for (auto ptr : allocations) {
    auto record = rm.findAllocationRecord(ptr);
    std::cout << "Allocation: " << record->ptr << ", Size: " << record->size << ", Name: " << record->name << std::endl;
  }

  //
  // Dump out all allocations for our allocator
  //
  std::stringstream ss;
  umpire::print_allocator_records(allocator, ss);
  std::cout << "Tracked allocators are: " << std::endl << ss.str() << std::endl;

  for (auto ptr : allocations) {
    allocator.deallocate(ptr);
  }
  return 0;
}
