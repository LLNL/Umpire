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
      "host_dynamic_pool", rm.getAllocator("HOST"), 512*sizeof(double), 256*sizeof(double));

  double* data = static_cast<double*>(
      alloc.allocate(256*sizeof(double)));

  for (std::size_t i = 0; i < 257; i++) {
    data[i] = 1.0 * i;
  }

  std::cout << "Data out of bounds (index 256)= " << data[256] << std::endl;
  std::cout << "Data out of bounds (index 511)= " << data[511] << std::endl;

  return 0;
}
