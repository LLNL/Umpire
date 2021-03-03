//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <chrono>
#include <string>
#include <random>

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/strategy/MixedPool.hpp"

#define CONVERT 1000000 //convert sec (s) to microsec (us)
#define ALLOCATIONS 1000000 //number of allocations for each round
#define NUM_ITER 3 //number of rounds (used to average timing)

const int size{1<<25}; //~33MB

/*
 * This functions measures the time it takes to do ALLOCATIONS allocations and 
 * then do ALLOCATIONS deallocations in the same order. The time is averaged across NUM_ITER rounds. 
 */
void same_order(umpire::Allocator alloc, std::string name)
{
  double time[2] = {0.0, 0.0};
  void* allocations[ALLOCATIONS];

  for(int i = 0; i < NUM_ITER; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < ALLOCATIONS; j++)
      allocations[j] = alloc.allocate(size);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/ALLOCATIONS;

    auto begin_dealloc = std::chrono::system_clock::now();
    for (int h = 0; h < ALLOCATIONS; h++)
      alloc.deallocate(allocations[h]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/ALLOCATIONS;
  }

  std::cout << "  SAME_ORDER (" << name << "):" << std::endl; 
  std::cout << "    alloc: " << (time[0]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl << std::endl;
}

/*
 * This functions measures the time it takes to do ALLOCATIONS allocations and 
 * then do ALLOCATIONS deallocations in reverse order. The time is averaged across NUM_ITER rounds. 
 */
void reverse_order(umpire::Allocator alloc, std::string name)
{
  double time[2] = {0.0, 0.0};
  void* allocations[ALLOCATIONS];

  for(int i = 0; i < NUM_ITER; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < ALLOCATIONS; j++)
      allocations[j] = alloc.allocate(size);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/ALLOCATIONS;

    auto begin_dealloc = std::chrono::system_clock::now();
    for (int h = (ALLOCATIONS-1); h >=0; h--)
      alloc.deallocate(allocations[h]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/ALLOCATIONS;
  }

  std::cout << "  REVERSE_ORDER (" << name << "):" << std::endl; 
  std::cout << "    alloc: " << (time[0]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl << std::endl;
}

/*
 * This functions measures the time it takes to do ALLOCATIONS allocations, shuffle the 
 * array of returned pointers, and then do ALLOCATIONS deallocations. The time is averaged
 * across NUM_ITER rounds. 
 */
void shuffle(umpire::Allocator alloc, std::string name)
{
  std::mt19937 gen(ALLOCATIONS);
  double time[2] = {0.0, 0.0};
  void* allocations[ALLOCATIONS];

  for(int i = 0; i < NUM_ITER; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < ALLOCATIONS; j++)
      allocations[j] = alloc.allocate(size);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/ALLOCATIONS;

    std::shuffle(&allocations[0], &allocations[ALLOCATIONS], gen);
    auto begin_dealloc = std::chrono::system_clock::now();
    for (int h = 0; h < ALLOCATIONS; h++)
      alloc.deallocate(allocations[h]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/ALLOCATIONS;
  }

  std::cout << "  SHUFFLE (" << name << "):" << std::endl; 
  std::cout << "    alloc: " << (time[0]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl << std::endl;
}

int main(int, char**) {
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9);

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");
  
  //FixedPool
  umpire::Allocator fixed_pool_alloc = rm.makeAllocator<
                    umpire::strategy::FixedPool, false>("fixed_pool", alloc, size);
  same_order(fixed_pool_alloc, "FixedPool");
  reverse_order(fixed_pool_alloc, "FixedPool");
  shuffle(fixed_pool_alloc, "FixedPool");

  //DynamicPoolMap
  umpire::Allocator dynamic_pool_map_alloc = rm.makeAllocator<
                    umpire::strategy::DynamicPoolMap, false>("dynamic_pool_map", alloc, size);
  same_order(dynamic_pool_map_alloc, "DynamicPoolMap");
  reverse_order(dynamic_pool_map_alloc, "DynamicPoolMap");
  shuffle(dynamic_pool_map_alloc, "DynamicPoolMap");

  //DynamicPooList
  umpire::Allocator dynamic_pool_list_alloc = rm.makeAllocator<
                    umpire::strategy::DynamicPoolList, false>("dynamic_pool_list", alloc, size);
  same_order(dynamic_pool_list_alloc, "DynamicPoolList");
  reverse_order(dynamic_pool_list_alloc, "DynamicPoolList");
  shuffle(dynamic_pool_list_alloc, "DynamicPoolList");

  //QuickPool
  umpire::Allocator quick_pool_alloc = rm.makeAllocator<
                    umpire::strategy::QuickPool, false>("quick_pool", alloc, size);
  same_order(quick_pool_alloc, "QuickPool");
  reverse_order(quick_pool_alloc, "QuickPool");
  shuffle(quick_pool_alloc, "QuickPool");

  //MixedPool
  umpire::Allocator mixed_pool_alloc = rm.makeAllocator<
                    umpire::strategy::MixedPool, false>("mixed_pool", alloc, size);
  same_order(mixed_pool_alloc, "MixedPool");
  reverse_order(mixed_pool_alloc, "MixedPool");
  shuffle(mixed_pool_alloc, "MixedPool");

  return 0;
}

