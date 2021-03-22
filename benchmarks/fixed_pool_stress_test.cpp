//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <chrono>
#include <string>
#include <random>
#include <numeric>

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

constexpr int CONVERT {1000000}; //convert sec (s) to microsec (us)
constexpr int NUM_RND {1000}; //number of rounds (used to average timing)
constexpr int NUM_ALLOC {512}; //number of allocations used for testing
constexpr int OBJECTS_PER_BLOCK {1<<11}; //number of blocks of object_bytes size (2048)

void test_deallocation_performance(umpire::Allocator alloc, int SIZE, std::vector<int> &indices, std::string test_name)
{
  double time[2] = {0.0, 0.0};
  void* allocations[NUM_ALLOC];

  for(int i = 0; i < NUM_RND; i++) {
    auto begin_alloc{std::chrono::system_clock::now()};
    for (int j = 0; j < NUM_ALLOC; j++)
      allocations[j] = alloc.allocate(SIZE);
    auto end_alloc{std::chrono::system_clock::now()};
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

  std::cout << "  " << test_name << " (FixedPool - " << (long long int)SIZE*OBJECTS_PER_BLOCK << "):" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

int main(int, char**)
{
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9); 

  auto& rm{umpire::ResourceManager::getInstance()};
  umpire::Allocator alloc{rm.getAllocator("HOST")};
  
  //Array of sizes used (large vs. medium vs. small)
  std::vector<int> sizes {67108864, 1048576, 2048};

  //create vector of indices for "same_order" tests
  std::vector<int> same_order_index(NUM_ALLOC);
  for(int i = 0; i < NUM_ALLOC; i++) {
    same_order_index[i] = i;
  }

  //create vector of indices for "reverse_order" tests
  std::vector<int> reverse_order_index(same_order_index);
  std::reverse(reverse_order_index.begin(), reverse_order_index.end());

  //create vector of indices for "shuffle_order" tests
  std::vector<int> shuffle_order_index(same_order_index);
  std::mt19937 gen(NUM_ALLOC);
  std::shuffle(&shuffle_order_index[0], &shuffle_order_index[NUM_ALLOC], gen);
 
  //create the FixedPool allocator and run stress tests for all sizes
  for(auto size : sizes)
  {
    umpire::Allocator pool_alloc = rm.makeAllocator<umpire::strategy::FixedPool, false>
                                 ("fixed_pool" + std::to_string(size), alloc, size, OBJECTS_PER_BLOCK);

    test_deallocation_performance(pool_alloc, size, same_order_index, "SAME_ORDER");
    test_deallocation_performance(pool_alloc, size, reverse_order_index, "REVERSE_ORDER");
    test_deallocation_performance(pool_alloc, size, shuffle_order_index, "SHUFFLE_ORDER");
  }

  return 0;
}
