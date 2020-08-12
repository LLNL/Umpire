//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/util/Macros.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto size_limited_alloc = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "size_limited_alloc", rm.getAllocator("HOST"), 1024);

  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "pool", size_limited_alloc, 64, 64);

  // This will throw an exception because the pool is limited to 1024 bytes.
  void* data = pool.allocate(2048);
  UMPIRE_USE_VAR(data);

  return 0;
}
