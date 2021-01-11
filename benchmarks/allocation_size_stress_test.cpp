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

#define ALLOCATIONS 1000
#define CONVERT 1000000 //convert sec (s) to microsec (us)
#define NUM_ITER 3

void benchmark_allocator_together(umpire::Allocator alloc, std::size_t size = 0) {
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

  std::cout << "  TOGETHER: " << std::endl;
  std::cout << "    alloc+dealloc: " <<  (time/double(NUM_ITER)*CONVERT) << "(us)" << std::endl;
}

void benchmark_allocator_separate(umpire::Allocator alloc, std::size_t size = 0) {
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
  auto& rm = umpire::ResourceManager::getInstance();

  //Begin calling the tests
  std::cout << "HOST" << std::endl;
  umpire::Allocator alloc = rm.getAllocator("HOST");
  benchmark_allocator_together(alloc);
  benchmark_allocator_separate(alloc);
#if defined(UMPIRE_ENABLE_DEVICE) 
  std::cout << "DEVICE" << std::endl;
  umpire::Allocator dalloc = rm.getAllocator("DEVICE");
  benchmark_allocator_together(dalloc);
  benchmark_allocator_separate(dalloc);
#endif
#if defined(UMPIRE_ENABLE_DEVICE_CONST) 
  std::cout << "DEVICE_CONST" << std::endl;
  umpire::Allocator dcalloc = rm.getAllocator("DEVICE_CONST");
  benchmark_allocator_together(dcalloc);
  benchmark_allocator_separate(dcalloc);
#endif
#if defined(UMPIRE_ENABLE_PINNED) 
  std::cout << "PINNED" << std::endl;
  umpire::Allocator palloc = rm.getAllocator("PINNED");
  benchmark_allocator_together(palloc);
  benchmark_allocator_separate(palloc);
#endif
#if defined(UMPIRE_ENABLE_UM) 
  std::cout << "UM" << std::endl;
  umpire::Allocator umalloc = rm.getAllocator("UM");
  benchmark_allocator_together(umalloc);
  benchmark_allocator_separate(umalloc);
#endif
#if defined(UMPIRE_ENABLE_FILE_RESOURCE) 
  std::cout << "FILE" << std::endl;
  umpire::Allocator falloc = rm.getAllocator("FILE");
  benchmark_allocator_together(falloc);
  benchmark_allocator_separate(falloc);
#endif

  return 0;
}
