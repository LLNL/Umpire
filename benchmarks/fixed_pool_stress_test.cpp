//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <chrono>
#include <string>
#include <random>
#include <numeric>
#include <map>

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/strategy/FixedPool.hpp"

constexpr std::size_t OBJECTS_PER_BLOCK {1<<11}; //number of blocks of object_bytes size (2048)

/*
 * \brief Function that tests the deallocation pattern performance of the FixedPool allocator. 
 *     The allocation, deallocation, and "lifespan" times are calculated and printed for each.
 * 
 * \param alloc, a given FixedPool allocator
 * \param size, number of bytes to allocate during timed test
 * \param indices, a vector of indices which are either structured in same_order, reverse_order, or
 *        shuffle_order, depending on the deallocation pattern being measured.
 * \param test_name, name of the deallocation pattern, used for output
 */
void test_deallocation_performance(umpire::Allocator alloc, std::size_t size, const std::vector<std::size_t> &indices, const std::string test_name)
{
  double time[] = {0.0, 0.0};
  constexpr std::size_t convert{1000000}; //convert sec (s) to microsec (us)
  constexpr std::size_t num_rnd{1000}; //number of rounds (used to average timing)
  const std::size_t num_indices{indices.size()};
  std::vector<void*> allocations(num_indices);

  for(std::size_t i{0}; i < num_rnd; i++) {
    auto begin_alloc{std::chrono::system_clock::now()};
    for (std::size_t j{0}; j < indices.size(); j++) {
      allocations[j] = alloc.allocate(size);
    }
    auto end_alloc{std::chrono::system_clock::now()};
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/num_indices;

    auto begin_dealloc {std::chrono::system_clock::now()};
    for (std::size_t h{0}; h < indices.size(); h++) {
      alloc.deallocate(allocations[indices[h]]);
    }
    auto end_dealloc {std::chrono::system_clock::now()};
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/num_indices;
  }

  alloc.release();
  double alloc_t{(time[0]/double(num_rnd)*convert)};
  double dealloc_t{(time[1]/double(num_rnd)*convert)};

  std::cout << "  " << test_name << " (FixedPool - " << (long long int)size*OBJECTS_PER_BLOCK << "):" << std::endl; 
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
  std::vector<std::size_t> sizes {67108864, 1048576, 2048};

  //number of allocations used for testing
  constexpr std::size_t num_alloc {512};
  std::mt19937 gen(num_alloc);

  //create map with name and vector of indices for tests
  std::map<const std::string, const std::vector<std::size_t>&> indexing_pairs;  
  std::vector<std::size_t> same_order(num_alloc);
  std::iota(same_order.begin(), same_order.end(), 0);

  std::vector<std::size_t> reverse_order(same_order.begin(), same_order.end());
  std::reverse(reverse_order.begin(), reverse_order.end());

  std::vector<std::size_t> shuffle_order(same_order.begin(), same_order.end());
  std::shuffle(shuffle_order.begin(), shuffle_order.end(), gen);

  //insert index vectors into map
  indexing_pairs.insert({"SAME_ORDER", same_order});
  indexing_pairs.insert({"REVERSE_ORDER", reverse_order});
  indexing_pairs.insert({"SHUFFLE_ORDER", shuffle_order});

  //create FixedPool allocator for each size and run test for each indexing pair
  for(auto size : sizes) {
    umpire::Allocator pool_alloc = rm.makeAllocator<umpire::strategy::FixedPool, false>
                 ("fixed_pool" + std::to_string(size), alloc, size, OBJECTS_PER_BLOCK);
    for(auto i : indexing_pairs) {
      test_deallocation_performance(pool_alloc, size, i.second, i.first);
    }
  }

  return 0;
}
