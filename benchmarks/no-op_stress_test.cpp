//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <chrono>
#include <random>
#include <map>
#include <numeric>

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#define CONVERT 1000000 //convert sec (s) to microsec (us)
#define ALLOCATIONS 1000000 //number of allocations for each round
#define NUM_ITER 3 //number of rounds (used to average timing)

const int size = 4096;

/*
 * \brief Function that tests the deallocation pattern performance of the no-op allocator. 
 *   The allocation, deallocation, and "lifespan" times are calculated and printed out for
 *   each allocator.
 *   
 * \param alloc, the no-op allocator
 * \param test_name, name of the deallocation pattern, used for output
 * \param indices, a vector of indices which are either structured in same_order, reverse_order, or
 *   shuffle_order, depending on the deallocation pattern being measured.
 */
void test_deallocation_performance(umpire::Allocator alloc, const std::string test_name, const std::vector<std::size_t> &indices)
{
  double time[2] = {0.0, 0.0};
  std::vector<void*> allocations(ALLOCATIONS);

  for(int i = 0; i < NUM_ITER; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (size_t j{0}; j < ALLOCATIONS; j++)
      allocations[j] = alloc.allocate(size);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/ALLOCATIONS;

    auto begin_dealloc = std::chrono::system_clock::now();
    for (size_t h{0}; h < ALLOCATIONS; h++)
      alloc.deallocate(allocations[indices[h]]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/ALLOCATIONS;
  }

  double alloc_t{(time[0]/double(NUM_ITER)*CONVERT)};
  double dealloc_t{(time[1]/double(NUM_ITER)*CONVERT)};

  std::cout << "  " << test_name << ":" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

int main(int, char**) {
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9);

  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("NO_OP");
  std::mt19937 gen(ALLOCATIONS);

  //create map with name and vector of indices for tests
  std::map<const std::string, const std::vector<std::size_t>&> indexing_pairs;  
  std::vector<std::size_t> same_order(ALLOCATIONS);
  std::iota(same_order.begin(), same_order.end(), 0);
  
  std::vector<std::size_t> reverse_order(same_order.begin(), same_order.end());
  std::reverse(reverse_order.begin(), reverse_order.end());
  
  std::vector<std::size_t> shuffle_order(same_order.begin(), same_order.end());
  std::shuffle(shuffle_order.begin(), shuffle_order.end(), gen);
  
  //insert indexing vectoring into map
  indexing_pairs.insert({"SAME_ORDER", same_order});
  indexing_pairs.insert({"REVERSE_ORDER", reverse_order});
  indexing_pairs.insert({"SHUFFLE_ORDER", shuffle_order});

  std::cout << " Testing allocating and deallocating " << std::endl
            << " with NO_OP resource: " << std::endl << std::endl;
 
  for(auto i : indexing_pairs) {
    test_deallocation_performance(alloc, i.first, i.second);
  }

  return 0;
}

