//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/util/Macros.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto no_op_alloc = rm.getAllocator("NO_OP");

  // allocations needed by the sample problem
  void* ptr = no_op_alloc.allocate(1024);
  no_op_alloc.deallocate(ptr);
  ////////////////////////////////////

  const int size_needed = no_op_alloc.getActualSize(); // Get total amount of memory used

  auto size_limited_alloc =
      rm.makeAllocator<umpire::strategy::SizeLimiter>("size_limited_alloc", rm.getAllocator("HOST"), size_needed);

  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool", size_limited_alloc, 64, 64);

  std::cout << "Attempting to allocate the exact amount of bytes needed for the sample problem..." << std::endl;
  try {
    void* data = pool.allocate(size_needed);
    UMPIRE_USE_VAR(data);
    std::cout << "SUCCESS!" << std::endl;
  } catch (...) {
    std::cout << "Exception caught! Trying to use more memory than this pool allows." << std::endl;
  }

  return 0;
}
