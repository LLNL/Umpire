//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/DynamicPool.hpp"

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_dynamic_pool", rm.getAllocator("HOST"));

  auto test = alloc.allocate(24);
  test = alloc.allocate(64);
  test = alloc.allocate(128);

  return (test == nullptr) ? 1 : 0;
}
