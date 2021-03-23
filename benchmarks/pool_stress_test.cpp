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

void test_deallocation_performance(umpire::Allocator alloc, std::vector<std::size_t> &indices, std::string test_name, std::string pool_name, std::size_t size)
{
  double time[] = {0.0, 0.0};
  constexpr int convert {1000000}; //convert sec (s) to microsec (us)
  constexpr int num_rnd {1000}; //number of rounds (used to average timing)
  std::size_t num_indices{indices.size()};
  std::vector<void*> allocations(num_indices);

  for(int i = 0; i < num_rnd; i++) {
    auto begin_alloc {std::chrono::system_clock::now()};
    for (std::size_t j{0}; j < num_indices; j++) {
      allocations[j] = alloc.allocate(size);
    }
    auto end_alloc {std::chrono::system_clock::now()};
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/num_indices;

    auto begin_dealloc {std::chrono::system_clock::now()};
    for (std::size_t h{0}; h < num_indices; h++) {
      alloc.deallocate(allocations[indices[h]]);
    }
    auto end_dealloc {std::chrono::system_clock::now()};
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/num_indices;
  }

  alloc.release();
  double alloc_t{(time[0]/double(num_rnd)*convert)};
  double dealloc_t{(time[1]/double(num_rnd)*convert)};

  std::cout << "  " << test_name <<" (" << pool_name << "):" << std::endl; 
  std::cout << "    alloc: " << alloc_t << "(us)" << std::endl;
  std::cout << "    dealloc: " << dealloc_t << "(us)" << std::endl;
  std::cout << "    lifetime: " << alloc_t+dealloc_t << "(us)" << std::endl << std::endl;
}

template <class T>
void do_test(std::string pool_name)
{
  //Set up test factors
  constexpr std::size_t alloc_size {137438953472ULL}; //137GiB, total size of all allocations together
  constexpr std::size_t size {268435456}; //268MiB, size of each allocation
  constexpr int num_alloc {alloc_size/size}; //number of allocations for each round

  auto& rm {umpire::ResourceManager::getInstance()};
  umpire::Allocator alloc {rm.getAllocator("HOST")};
  umpire::Allocator pool_alloc {rm.makeAllocator<T, false>(pool_name, alloc, alloc_size)};

  //create vector of indices for "same_order" tests
  std::vector<int> ordering_index;
  for(int i = 0; i < num_alloc; i++) {
    ordering_index.push_back(i);
  }
  test_deallocation_performance(pool_alloc, ordering_index, "SAME_ORDER", pool_name, size);
  
  //create vector of indices for "reverse_order" tests
  std::reverse(ordering_index.begin(), ordering_index.end());
  test_deallocation_performance(pool_alloc, ordering_index, "REVERSE_ORDER", pool_name, size);
  
  //create vector of indices for "shuffle_order" tests
  std::mt19937 gen(num_alloc);
  std::shuffle(ordering_index.begin(), ordering_index.end(), gen);
  test_deallocation_performance(pool_alloc, ordering_index, "SHUFFLE_ORDER", pool_name, size);
}

int main(int, char**) {
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9);

  //Call template function to run tests for each pool
  do_test<umpire::strategy::DynamicPoolMap>("DynamicPoolMap");
  do_test<umpire::strategy::DynamicPoolList>("DynamicPoolList");
  do_test<umpire::strategy::QuickPool>("QuickPool");
  do_test<umpire::strategy::MixedPool>("MixedPool");

  return 0;
}
