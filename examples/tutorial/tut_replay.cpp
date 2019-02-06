//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include <functional>
#include <random>

int main(int, char**) {
  constexpr int NUM_ALLOCATIONS = 64;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, 1024);

  auto random_number = std::bind(distribution, generator);

  std::vector<void*> allocations;

  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("HOST");

  // Make an allocator:
  auto pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "pool",
      allocator);

  // Do some allocations:
  for ( int i = 0; i < NUM_ALLOCATIONS; ++i ) {
    allocations.push_back( pool.allocate(random_number()));
  }

  for ( auto ptr : allocations ) {
    pool.deallocate(ptr);
  }

  return 0;
}
