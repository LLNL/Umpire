//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <chrono>

#include <random>

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#define CONVERT 1000000 //convert sec (s) to microsec (us)
#define ALLOCATIONS 1000000 //number of allocations for each round
#define NUM_ITER 3 //number of rounds (used to average timing)

const int size = 4096;

/*
 * This functions measures the time it takes to do ALLOCATIONS no-op allocations and 
 * then do ALLOCATIONS no-op deallocations in the same order. The time is averaged across 3 rounds. 
 */
void same_order(umpire::Allocator alloc)
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

  std::cout << "  SAME_ORDER:" << std::endl; 
  std::cout << "    alloc: " << (time[0]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl << std::endl;
}

/*
 * This functions measures the time it takes to do ALLOCATIONS no-op allocations and 
 * then do ALLOCATIONS no-op deallocations in reverse order. The time is averaged across 3 rounds. 
 */
void reverse_order(umpire::Allocator alloc)
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

  std::cout << "  REVERSE_ORDER:" << std::endl; 
  std::cout << "    alloc: " << (time[0]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl << std::endl;
}

/*
 * This functions measures the time it takes to do ALLOCATIONS no-op allocations, shuffle the 
 * array of returned pointers, and then do ALLOCATIONS no-op deallocations. The time is averaged
 * across 3 rounds. 
 */
void shuffle(umpire::Allocator alloc)
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

  std::cout << "  SHUFFLE:" << std::endl; 
  std::cout << "    alloc: " << (time[0]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl << std::endl;
}

int main(int, char**) {
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9);

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("NO_OP");

  std::cout << " Testing allocating and deallocating " << std::endl
            << " with NO_OP resource: " << std::endl << std::endl;
  same_order(alloc);
  reverse_order(alloc);
  shuffle(alloc);

  return 0;
}

