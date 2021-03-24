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

/*
 * \brief Function that tests the deallocation pattern performance of a given pool allocator. 
 *       The allocation, deallocation, and "lifespan" times are calculated and printed out for
 *       each allocator.
 * 
 * \param alloc, a given pool allocator
 * \param indices, a vector of indices which are either structured in same_order, reverse_order, or
 *        shuffle_order, depending on the deallocation pattern being measured.
 * \param test_name, name of the deallocation pattern, used for output
 * \param pool_name, name of the pool strategy, used for output
 * \param size, number of bytes to allocate during timed test
 */
void test_deallocation_performance(umpire::Allocator alloc, std::string pool_name, std::vector<std::size_t> &indices, std::string test_name, std::size_t size)
{
  double time[] = {0.0, 0.0};
  constexpr size_t convert {1000000}; //convert sec (s) to microsec (us)
  constexpr size_t num_rnd {1000}; //number of rounds (used to average timing)
  std::size_t num_indices{indices.size()};
  std::vector<void*> allocations(num_indices);

  for(std::size_t i{0}; i < num_rnd; i++) {
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
void do_test(std::string pool_name, std::map<std::string, std::vector<std::size_t>> indexing_pairs, size_t alloc_size, size_t size)
{
  auto& rm {umpire::ResourceManager::getInstance()};
  umpire::Allocator alloc {rm.getAllocator("HOST")};
  umpire::Allocator pool_alloc {rm.makeAllocator<T, false>(pool_name, alloc, alloc_size)};

  for(auto i : indexing_pairs)
    test_deallocation_performance(pool_alloc, pool_name, i.second, i.first, size);
}

int main(int, char**) {
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9);

  //Set up test factors
  constexpr std::size_t alloc_size {137438953472ULL}; //137GiB, total size of all allocations together
  constexpr std::size_t size {268435456}; //268MiB, size of each allocation
  constexpr std::size_t num_alloc {alloc_size/size}; //number of allocations for each round
  std::mt19937 gen(num_alloc);

  //create vector of indices for tests
  std::map<std::string, std::vector<std::size_t>> indexing_pairs;  
  for(std::size_t i{0}; i < num_alloc; i++) {
    indexing_pairs["SAME_ORDER"].push_back(i);
    indexing_pairs["REVERSE_ORDER"].push_back(i);
    indexing_pairs["SHUFFLE_ORDER"].push_back(i);
  }
  std::reverse(indexing_pairs["REVERSE_ORDER"].begin(), indexing_pairs["REVERSE_ORDER"].end());
  std::shuffle(indexing_pairs["SHUFFLE_ORDER"].begin(), indexing_pairs["SHUFFLE_ORDER"].end(), gen);

  //Call template function to run tests for each pool
  do_test<umpire::strategy::DynamicPoolMap> ("DynamicPoolMap", indexing_pairs, alloc_size, size);
  do_test<umpire::strategy::DynamicPoolList> ("DynamicPoolList", indexing_pairs, alloc_size, size);
  do_test<umpire::strategy::QuickPool> ("QuickPool", indexing_pairs, alloc_size, size);
  do_test<umpire::strategy::MixedPool> ("MixedPool", indexing_pairs, alloc_size, size);

  return 0;
}
