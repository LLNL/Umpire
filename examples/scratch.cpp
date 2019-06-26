//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_simpool", rm.getAllocator("HOST"));


  char* data = (char*) alloc.allocate(0);

  char* real = (char*) alloc.allocate(1024);

  std::cout << "0 ptr: " << static_cast<void*>(data) << std::endl;
  std::cout << "real ptr: " << static_cast<void*>(real) << std::endl;

  alloc.deallocate(real);

  std::cout << "Accessing 0 ptr... " << std::endl;
  std::cout << data[0] << std::endl;

  return 0;
}
