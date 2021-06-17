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
#define ALLOCATIONS 10000 //number of allocations for each round
#define NUM_ITER 1000 //number of rounds (used to average timing)

//since no memory is actually being allocated, this value will only be used
//to increment a counter in the No Op resource
const int size = 4096;

/*
 * This functions measures the time it takes to do ALLOCATIONS no-op allocations and 
 * then do ALLOCATIONS no-op deallocations in the same order. The time is averaged across NUM_ITER rounds. 
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

  double alloc_t{(time[0]/double(NUM_ITER)*CONVERT)};
  double dealloc_t{(time[1]/double(NUM_ITER)*CONVERT)};

  std::cout << "  SAME_ORDER:" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

/*
 * This functions measures the time it takes to do ALLOCATIONS no-op allocations and 
 * then do ALLOCATIONS no-op deallocations in reverse order. The time is averaged across NUM_ITER rounds. 
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

  double alloc_t{(time[0]/double(NUM_ITER)*CONVERT)};
  double dealloc_t{(time[1]/double(NUM_ITER)*CONVERT)};

  std::cout << "  REVERSE_ORDER:" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

/*
 * This functions measures the time it takes to do ALLOCATIONS no-op allocations, shuffle the 
 * array of returned pointers, and then do ALLOCATIONS no-op deallocations. The time is averaged
 * across NUM_ITER rounds. 
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

  double alloc_t{(time[0]/double(NUM_ITER)*CONVERT)};
  double dealloc_t{(time[1]/double(NUM_ITER)*CONVERT)};

  std::cout << "  SHUFFLE:" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

int main(int, char**) {
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9);
  std::cout << "Total number of allocations: " << ALLOCATIONS << std::endl;
  std::cout << "(Both the size per allocation and total amount of memory " 
            << "allocated do not matter since this is the No-Op benchmark)"
            << std::endl;

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("NO_OP");

  std::cout << " Testing allocating and deallocating " << std::endl
            << " with NO_OP resource: " << std::endl << std::endl;
  same_order(alloc);
  reverse_order(alloc);
  shuffle(alloc);

  return 0;
}

