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

#if defined (UMPIRE_ENABLE_CUDA)
  constexpr std::size_t ALLOC_SIZE {8589934592ULL}; //8GiB total size of all allocations together
#else
  constexpr std::size_t ALLOC_SIZE {137438953472ULL}; //137GiB total size of all allocations together
#endif

constexpr std::size_t SIZE {268435456}; //268MiB, size of each allocation

/*
 * \brief Function that tests the deallocation pattern performance of a given pool allocator. 
 *       The allocation, deallocation, and "lifespan" times are calculated and printed out for
 *       each allocator.
 * 
 * \param alloc, a given pool allocator
 * \param pool_name, name of the pool strategy, used for output
 * \param indices, a vector of indices which are either structured in same_order, reverse_order, or
 *        shuffle_order, depending on the deallocation pattern being measured.
 * \param test_name, name of the deallocation pattern, used for output
 */
void test_deallocation_performance(umpire::Allocator alloc, std::string pool_name, const std::vector<std::size_t> &indices, const std::string test_name)
{
  double time[] = {0.0, 0.0};
  constexpr std::size_t convert {1000000}; //convert sec (s) to microsec (us)
  constexpr std::size_t num_rnd {1000}; //number of rounds (used to average timing)
  const std::size_t num_indices{indices.size()};
  std::vector<void*> allocations(num_indices);

  for(std::size_t i{0}; i < num_rnd; i++) {
    auto begin_alloc {std::chrono::system_clock::now()};
    for (std::size_t j{0}; j < num_indices; j++) {
      allocations[j] = alloc.allocate(SIZE);
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
void do_test(std::string pool_name, std::map<const std::string, const std::vector<std::size_t>&> indexing_pairs)
{
  auto& rm {umpire::ResourceManager::getInstance()};
  umpire::Allocator alloc {rm.getAllocator("DEVICE")};
  umpire::Allocator pool_alloc {rm.makeAllocator<T, false>(pool_name, alloc, ALLOC_SIZE)};

  for(auto i : indexing_pairs) {
    test_deallocation_performance(pool_alloc, pool_name, i.second, i.first);
  }
}

int main(int, char**) {
  //Set up formatting for output
  std::cout << std::fixed << std::setprecision(9);

  //Set up test factors
  constexpr std::size_t num_alloc {ALLOC_SIZE/SIZE}; //number of allocations for each round
  std::mt19937 gen(num_alloc);

  //create map with name and vector of indices for tests
  std::map<const std::string, const std::vector<std::size_t>&> indexing_pairs;  
  std::vector<std::size_t> same_order(num_alloc);
  std::iota(same_order.begin(), same_order.end(), 0);

  std::vector<std::size_t> reverse_order(same_order.begin(), same_order.end());
  std::reverse(reverse_order.begin(), reverse_order.end());

  std::vector<std::size_t> shuffle_order(same_order.begin(), same_order.end());
  std::shuffle(shuffle_order.begin(), shuffle_order.end(), gen);

  //insert indexing vectoring into map
  indexing_pairs.insert({"SAME_ORDER", same_order});
  indexing_pairs.insert({"REVERSE_ORDER", reverse_order});
  indexing_pairs.insert({"SHUFFLE_ORDER", shuffle_order});

  //Call template function to run tests for each pool
  do_test<umpire::strategy::DynamicPoolMap> ("DynamicPoolMap", indexing_pairs);
  do_test<umpire::strategy::DynamicPoolList> ("DynamicPoolList", indexing_pairs);
  do_test<umpire::strategy::QuickPool> ("QuickPool", indexing_pairs);
  do_test<umpire::strategy::MixedPool> ("MixedPool", indexing_pairs);

  return 0;
}
