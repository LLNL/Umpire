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
#include "umpire/Umpire.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto size_limited_alloc =
      rm.makeAllocator<umpire::strategy::SizeLimiter>("size_limited_alloc", rm.getAllocator("HOST"), 1024);

  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool", size_limited_alloc, 64, 64);
  void* data;

  // This will throw an exception because the pool is limited to 1024 bytes.
  std::cout << "Attempting to allocate 2098 bytes..." << std::endl;
  try {
    data = pool.allocate(2048);
    UMPIRE_USE_VAR(data);
  } catch (...) {
    std::cout << "Exception caught! Pool is limited to 1024 bytes." << std::endl;
  }
  std::cout << "The total amount of memory used was: " << umpire::get_total_memory_allocated() << std::endl;

  std::cout << "Attempting to allocate 512 bytes..." << std::endl;
  try {
    data = pool.allocate(512);
    UMPIRE_USE_VAR(data);
  } catch (...) {
    std::cout << "Exception caught! Pool is limited to 1024 bytes." << std::endl;
  }
  std::cout << "The total amount of memory used was: " << umpire::get_total_memory_allocated() << std::endl;

  if (data != nullptr) {
    pool.deallocate(data);
  }

  return 0;
}
