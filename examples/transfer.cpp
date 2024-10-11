//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/NamedAllocationStrategy.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc_one = rm.makeAllocator<umpire::strategy::NamedAllocationStrategy>("named_one", rm.getAllocator("HOST"));
  auto alloc_two = rm.makeAllocator<umpire::strategy::NamedAllocationStrategy>("named_two", rm.getAllocator("HOST"));

  void* test = alloc_one.allocate(100);

  rm.transfer(test, alloc_two);

  alloc_two.deallocate(test);

  return 0;
}
