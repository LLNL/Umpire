//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <functional>
#include <random>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

int main(int, char**)
{
  constexpr int NUM_ALLOCATIONS = 64;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution{0, 1024};

  auto random_number = std::bind(distribution, generator);

  std::vector<void*> allocations{NUM_ALLOCATIONS, nullptr};

  auto& rm = umpire::ResourceManager::getInstance();

  // Make an allocator
  auto allocator = rm.getAllocator("HOST");
  auto pool =
      rm.makeAllocator<umpire::strategy::DynamicPool>("pool", allocator);

  // Do some allocations
  std::generate(allocations.begin(), allocations.end(),
                [&]() { return pool.allocate(random_number()); });

  // Clean up
  for (auto& ptr : allocations)
    pool.deallocate(ptr);

  return 0;
}
