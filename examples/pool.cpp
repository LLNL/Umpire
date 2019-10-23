//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/Pool.hpp"

#include <iostream>
#include <random>

int main() {
  //constexpr int ALLOCATIONS{1024};

  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::Pool>(
      "POOL", rm.getAllocator("HOST"), 512, 128);

  std::vector<void*> data;

  data.push_back(alloc.allocate(128));
  data.push_back(alloc.allocate(128));
  data.push_back(alloc.allocate(128));

  alloc.deallocate(data[1]);
  alloc.deallocate(data[2]);

  data.push_back(alloc.allocate(256));

  alloc.deallocate(data[0]);
  alloc.deallocate(data[3]);

  alloc.release();

  // std::mt19937 gen(12345678);
  // std::uniform_int_distribution<std::size_t> dist(64, 4096);

  // void* allocations[ALLOCATIONS];

  // for (int i = 0; i < ALLOCATIONS; i++) {
  //   std::size_t size = dist(gen);
  //   allocations[i] = alloc.allocate(size);
  // }
  // for (int i = 0; i < ALLOCATIONS; i++) {
  //   alloc.deallocate(allocations[i]);
  // }

  return 0;
}
