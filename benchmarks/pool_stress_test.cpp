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
#include <numeric>

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/strategy/MixedPool.hpp"

#define CONVERT 1000000 //convert sec (s) to microsec (us)

//Set up test factors
const long long unsigned int ALLOC_SIZE {137438953472}; //137GiB, total size of all allocations together
const int SIZE {1<<28}; //268MiB, size of each allocation
const int NUM_ALLOC {512}; //number of allocations for each round
const int NUM_RND {1000}; //number of rounds (used to average timing)

void run_test(umpire::Allocator alloc, std::vector<int> indices, std::string test_name, std::string pool_name)
{
  double time[2] = {0.0, 0.0};
  void* allocations[NUM_ALLOC];

  for(int i = 0; i < NUM_RND; i++) {
    auto begin_alloc {std::chrono::system_clock::now()};
    for (int j = 0; j < NUM_ALLOC; j++)
      allocations[indices[j]] = alloc.allocate(SIZE);
    auto end_alloc {std::chrono::system_clock::now()};
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/NUM_ALLOC;

    auto begin_dealloc {std::chrono::system_clock::now()};
    for (int h = 0; h < NUM_ALLOC; h++)
      alloc.deallocate(allocations[indices[h]]);
    auto end_dealloc {std::chrono::system_clock::now()};
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/NUM_ALLOC;
  }

  alloc.release();
  double alloc_t{(time[0]/double(NUM_RND)*CONVERT)};
  double dealloc_t{(time[1]/double(NUM_RND)*CONVERT)};

  std::cout << "  " << test_name <<" (" << pool_name << "):" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

template <class T>
void setup_test(std::string pool_name)
{
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");
  umpire::Allocator pool_alloc = rm.makeAllocator<T, false>(pool_name, alloc, ALLOC_SIZE);

  //create vector of indices for "same_order" tests
  std::vector<int> same_order_index(NUM_ALLOC);
  for(int i = 0; i < NUM_ALLOC; i++) 
    same_order_index[i] = i;
  
  //create vector of indices for "reverse_order" tests
  std::vector<int> reverse_order_index(same_order_index);
  std::reverse(reverse_order_index.begin(), reverse_order_index.end());
  
  //create vector of indices for "shuffle_order" tests
  std::vector<int> shuffle_order_index(same_order_index);
  std::mt19937 gen(NUM_ALLOC);
  std::shuffle(shuffle_order_index.begin(), shuffle_order_index.end(), gen);

  run_test(pool_alloc, same_order_index, "SAME_ORDER", pool_name);
  run_test(pool_alloc, reverse_order_index, "REVERSE_ORDER", pool_name);
  run_test(pool_alloc, shuffle_order_index, "SHUFFLE_ORDER", pool_name);
}

int main(int, char**) {
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9);

  //Call template function to run tests for each pool
  setup_test<umpire::strategy::DynamicPoolMap>("DynamicPoolMap");
  setup_test<umpire::strategy::DynamicPoolList>("DynamicPoolList");
  setup_test<umpire::strategy::QuickPool>("QuickPool");
  setup_test<umpire::strategy::MixedPool>("MixedPool");

  return 0;
}

