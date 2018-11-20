//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/DynamicPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto size_limited_alloc = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "size_limited_alloc", rm.getAllocator("HOST"), 1024);

  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "pool", size_limited_alloc, 64, 64);

  // This will throw an exception because the pool is limited to 1024 bytes.
  void* data = pool.allocate(2048);

  return 0;
}
