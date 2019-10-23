//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/Pool.hpp"

#include <iostream>

int main() {
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::Pool>(
      "POOL", rm.getAllocator("HOST"));

  void* data = alloc.allocate(1024);

  alloc.deallocate(data);

  return 0;
}
