//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/QuickPool.hpp"

// NOTE:
// To test with ASAN, you must reconfigure Umpire with
// CMAKE_CXX_FLAGS="-fsanitize=address"
//
int main(int, char**)
{
  const bool try_list{false};
  const bool try_quick{true};
  double* data{nullptr};
  auto& rm = umpire::ResourceManager::getInstance();

  if (try_list) {
    auto pool = rm.makeAllocator<umpire::strategy::DynamicPoolList>("pool", rm.getAllocator("HOST"));
    data = static_cast<double*>(pool.allocate(1024 * sizeof(double)));
    data[1023] = 100;
    data[1024] = 100;
    pool.deallocate(data);
  } else if (try_quick) {
    auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool", rm.getAllocator("HOST"));
    data = static_cast<double*>(pool.allocate(1024 * sizeof(double)));
    data[1023] = 100;
    data[1024] = 100;
    pool.deallocate(data);
  } else {
    std::cout << "No pools to try" << std::endl;
    return 1;
  }

  std::cout << "data[0] = " << data[0] << std::endl;

  return 0;
}
