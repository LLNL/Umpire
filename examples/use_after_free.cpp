//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool", rm.getAllocator("HOST"));

  double* data = static_cast<double*>(pool.allocate(1024 * sizeof(double)));
  data[256] = 100;
  std::cout << "data[256] = " << data[256] << std::endl;
  pool.deallocate(data);
  std::cout << "data[256] = " << data[256] << std::endl;

  return 0;
}
