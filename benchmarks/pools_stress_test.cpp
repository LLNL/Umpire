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
#define NUM_RND 100 //number of rounds (used to average timing)

const unsigned long long int ALLOC_SIZE = 17179869184; //~17GB - total SIZE of all allocations together
const int SIZE{1<<25}; //~33MB - size of each allocation
const int NUM_ALLOC = ALLOC_SIZE/SIZE; //number of allocations for each round

/*
 * This functions measures the time it takes to do NUM_ALLOC allocations and 
 * then do NUM_ALLOC deallocations in the same order. The time is averaged across NUM_RND rounds. 
 */
void same_order(umpire::Allocator alloc, std::string name)
{
  double time[2] = {0.0, 0.0};
  void* allocations[NUM_ALLOC];

  for(int i = 0; i < NUM_RND; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < NUM_ALLOC; j++)
      allocations[j] = alloc.allocate(SIZE);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/NUM_ALLOC;

    auto begin_dealloc = std::chrono::system_clock::now();
    for (int h = 0; h < NUM_ALLOC; h++)
      alloc.deallocate(allocations[h]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/NUM_ALLOC;
  }

  alloc.release();

  std::cout << "  SAME_ORDER (" << name << "):" << std::endl; 
  std::cout << "    alloc: " << (time[0]/double(NUM_RND)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_RND)*CONVERT) << "(us)" << std::endl << std::endl;
}

/*
 * This functions measures the time it takes to do NUM_ALLOC allocations and 
 * then do NUM_ALLOC deallocations in reverse order. The time is averaged across NUM_RND rounds. 
 */
void reverse_order(umpire::Allocator alloc, std::string name)
{
  double time[2] = {0.0, 0.0};
  void* allocations[NUM_ALLOC];

  for(int i = 0; i < NUM_RND; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < NUM_ALLOC; j++)
      allocations[j] = alloc.allocate(SIZE);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/NUM_ALLOC;

    auto begin_dealloc = std::chrono::system_clock::now();
    for (int h = (NUM_ALLOC-1); h >=0; h--)
      alloc.deallocate(allocations[h]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/NUM_ALLOC;
  }

  alloc.release();

  std::cout << "  REVERSE_ORDER (" << name << "):" << std::endl; 
  std::cout << "    alloc: " << (time[0]/double(NUM_RND)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_RND)*CONVERT) << "(us)" << std::endl << std::endl;
}

/*
 * This functions measures the time it takes to do NUM_ALLOC allocations, shuffle the 
 * array of returned pointers, and then do NUM_ALLOC deallocations. The time is averaged
 * across NUM_RND rounds. 
 */
void shuffle(umpire::Allocator alloc, std::string name)
{
  std::mt19937 gen(NUM_ALLOC);
  double time[2] = {0.0, 0.0};
  void* allocations[NUM_ALLOC];

  for(int i = 0; i < NUM_RND; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < NUM_ALLOC; j++)
      allocations[j] = alloc.allocate(SIZE);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/NUM_ALLOC;

    std::shuffle(&allocations[0], &allocations[NUM_ALLOC], gen);
    auto begin_dealloc = std::chrono::system_clock::now();
    for (int h = 0; h < NUM_ALLOC; h++)
      alloc.deallocate(allocations[h]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/NUM_ALLOC;
  }

  alloc.release();

  std::cout << "  SHUFFLE (" << name << "):" << std::endl; 
  std::cout << "    alloc: " << (time[0]/double(NUM_RND)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_RND)*CONVERT) << "(us)" << std::endl << std::endl;
}

int main(int, char**) {
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9);

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");
  
  //FixedPool
  umpire::Allocator fixed_pool_alloc = rm.makeAllocator<
                    umpire::strategy::FixedPool, false>("fixed_pool", alloc, SIZE);
  same_order(fixed_pool_alloc, "FixedPool");
  reverse_order(fixed_pool_alloc, "FixedPool");
  shuffle(fixed_pool_alloc, "FixedPool");

  //DynamicPoolMap
  umpire::Allocator dynamic_pool_map_alloc = rm.makeAllocator<
                    umpire::strategy::DynamicPoolMap, false>("dynamic_pool_map", alloc, SIZE);
  same_order(dynamic_pool_map_alloc, "DynamicPoolMap");
  reverse_order(dynamic_pool_map_alloc, "DynamicPoolMap");
  shuffle(dynamic_pool_map_alloc, "DynamicPoolMap");

  //DynamicPooList
  umpire::Allocator dynamic_pool_list_alloc = rm.makeAllocator<
                    umpire::strategy::DynamicPoolList, false>("dynamic_pool_list", alloc, SIZE);
  same_order(dynamic_pool_list_alloc, "DynamicPoolList");
  reverse_order(dynamic_pool_list_alloc, "DynamicPoolList");
  shuffle(dynamic_pool_list_alloc, "DynamicPoolList");

  //QuickPool
  umpire::Allocator quick_pool_alloc = rm.makeAllocator<
                    umpire::strategy::QuickPool, false>("quick_pool", alloc, SIZE);
  same_order(quick_pool_alloc, "QuickPool");
  reverse_order(quick_pool_alloc, "QuickPool");
  shuffle(quick_pool_alloc, "QuickPool");

  //MixedPool
  umpire::Allocator mixed_pool_alloc = rm.makeAllocator<
                    umpire::strategy::MixedPool, false>("mixed_pool", alloc, SIZE);
  same_order(mixed_pool_alloc, "MixedPool");
  reverse_order(mixed_pool_alloc, "MixedPool");
  shuffle(mixed_pool_alloc, "MixedPool");

  return 0;
}

