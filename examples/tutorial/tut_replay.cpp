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
  // _sphinx_tag_tut_replay_make_allocate_start
  auto allocator = rm.getAllocator("HOST");
  auto pool =
      rm.makeAllocator<umpire::strategy::DynamicPool>("pool", allocator);
  // _sphinx_tag_tut_replay_make_allocate_end

  // Do some allocations
  // _sphinx_tag_tut_replay_allocate_start
  std::generate(allocations.begin(), allocations.end(),
                [&]() { return pool.allocate(random_number()); });
  // _sphinx_tag_tut_replay_allocate_end

  // Clean up
  // _sphinx_tag_tut_replay_dealocate_start
  for (auto& ptr : allocations)
    pool.deallocate(ptr);
  // _sphinx_tag_tut_replay_dealocate_end

  return 0;
}
