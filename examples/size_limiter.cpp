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
  auto size_limited_alloc =
      rm.makeAllocator<umpire::strategy::SizeLimiter>("size_limited_alloc", rm.getAllocator("HOST"), 1024);

  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool", size_limited_alloc, 64, 64);

  // This will throw an exception because the pool is limited to 1024 bytes.
  std::cout << "Attempting to allocate 2098 bytes..." << std::endl;
  try {
    void* data = pool.allocate(2048);
    UMPIRE_USE_VAR(data);
  } catch (...) {
    std::cout << "Exception caught! Pool is limited to 1024 bytes." << std::endl;
  }

  return 0;
}
