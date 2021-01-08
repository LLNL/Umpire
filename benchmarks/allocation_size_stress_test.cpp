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

#define ALLOCATIONS 1000000
#define CONVERT 1000000 //convert sec (s) to microsec (us)
#define NUM_ITER 3

void benchmark_allocator_one(std::string name, std::size_t size = 0) {
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator(name);
  void* allocations[ALLOCATIONS];
  double time = 0.0;

  if (size == 0) {
    std::mt19937 gen(12345678);
    std::uniform_int_distribution<std::size_t> dist(64, 4096);
    size = dist(gen);
  }  

  for(int i = 0; i < NUM_ITER; i++) {
    auto begin_all_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < ALLOCATIONS; j++) {
      allocations[j] = alloc.allocate(size);
      alloc.deallocate(allocations[j]);
    }
    auto end_all_dealloc = std::chrono::system_clock::now();
    time += std::chrono::duration<double>(end_all_dealloc - begin_all_alloc).count()/ALLOCATIONS;
  }

  std::cout << name << std::endl;
  std::cout << "  TOGETHER: " << std::endl;
  std::cout << "    alloc+dealloc: " <<  (time/double(NUM_ITER)*CONVERT) << "(us)" << std::endl;
}

void benchmark_allocator_two(std::string name, std::size_t size = 0) {
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator(name);
  void* allocations[ALLOCATIONS];
  double time[2] = {0.0, 0.0};

  if (size == 0) {
    std::mt19937 gen(12345678);
    std::uniform_int_distribution<std::size_t> dist(64, 4096);
    size = dist(gen);
  }  

  for(int i = 0; i < NUM_ITER; i++) {
    auto begin_alloc = std::chrono::system_clock::now();
    for (int j = 0; j < ALLOCATIONS; j++) 
      allocations[j] = alloc.allocate(size);
    auto end_alloc = std::chrono::system_clock::now();
    time[0] += std::chrono::duration<double>(end_alloc - begin_alloc).count()/ALLOCATIONS;
    
    auto begin_dealloc = std::chrono::system_clock::now();
    for (int j = 0; j < ALLOCATIONS; j++) 
      alloc.deallocate(allocations[j]);
    auto end_dealloc = std::chrono::system_clock::now();
    time[1] += std::chrono::duration<double>(end_dealloc - begin_dealloc).count()/ALLOCATIONS;
  }

  std::cout << "  SEPARATE:   " << std::endl;
  std::cout << "    alloc: " << (time[0]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl;
  std::cout << "    dealloc: " << (time[1]/double(NUM_ITER)*CONVERT) << "(us)" << std::endl << std::endl;
}

int main(int, char**) {
  //Set the output format
  std::cout << std::fixed << std::setprecision(9);

  //Begin calling the tests
  benchmark_allocator_one("HOST");
  benchmark_allocator_two("HOST");
#if defined(UMPIRE_ENABLE_DEVICE) 
  benchmark_allocator_one("DEVICE");
  benchmark_allocator_two("DEVICE");
#endif
#if defined(UMPIRE_ENABLE_PINNED) 
  benchmark_allocator_one("PINNED");
  benchmark_allocator_two("PINNED");
#endif
#if defined(UMPIRE_ENABLE_UM) 
  benchmark_allocator_one("UM");
  benchmark_allocator_two("UM");
#endif

  return 0;
}
