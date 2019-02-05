
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

#include "umpire/strategy/MixedPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::MixedPool>(
      "host_mixed_pool", rm.getAllocator("HOST"));

  pool.allocate(62);
  pool.allocate(120);
  pool.allocate(245);
  pool.allocate(489);

  return 0;
}
