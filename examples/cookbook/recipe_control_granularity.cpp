//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/GranularityController.hpp"
// #include "umpire/strategy/QuickPool.hpp"

void use_allocator(const std::string& resource)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator(resource);

  auto alloc = rm.makeAllocator<umpire::strategy::GranularityController>(
    resource + "_COARSE", 
    allocator, 
    umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence);
  
  std::vector<void*> ptrs;
  const int N{10};
  int size{2};

  for ( int i = 0; i < N; i++) {
    ptrs.push_back(alloc.allocate(size));
    std::cout << ptrs[i] << std::endl;
    size *= 2;
  }

  for ( int i = 0; i < N; i++) {
    alloc.deallocate(ptrs[i]);
  }
}

int main(int, char**)
{
  // const std::vector<std::string> resources{"UM", "DEVICE_CONST", "DEVICE", "PINNED"};
  const std::vector<std::string> resources{"DEVICE"};

  for ( auto& resource : resources ) {
    use_allocator(resource);
  }
  return 0;
}
