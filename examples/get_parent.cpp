//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

void check_parent(umpire::Allocator alloc)
{
  umpire::strategy::AllocationStrategy* root = alloc.getAllocationStrategy();
  while ((root->getParent() != nullptr)) {
    root = root->getParent();
    std::cout << root->getName() << std::endl;
  }
  //
  // At this point, root will be pointing to the resource being used.
  //
  std::cout << "(root allocator)" << std::endl;
  std::cout << "------------------" << std::endl;
}

/*
 * This example exposes the parent allocator or resource that was used
 * when creating a new allocator. The first parent will be NULL because
 * this represents the memory resource being used. From that point on,
 * the parent will be whatever allocator was used by the resource manager
 * to create the next allocator in the chain.
 */
int main()
{
  const int SIZE = 5;
  auto& rm = umpire::ResourceManager::getInstance();

  std::vector<umpire::Allocator> Alloc(SIZE);
  Alloc[0] = rm.getAllocator("HOST");

  for (int i = 1; i < SIZE; i++) {
    Alloc[i] = rm.makeAllocator<umpire::strategy::QuickPool>("HOST_pool_" + std::to_string(i), Alloc[i - 1]);
  }

  for (int c = 0; c < SIZE; c++) {
    std::cout << "Starting tree: " << (c + 1) << "/" << SIZE << std::endl;
    check_parent(Alloc[c]);
  }

  return 0;
}
