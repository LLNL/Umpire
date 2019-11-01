//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/Pool.hpp"

#include <iostream>
#include <random>
#include <chrono>

constexpr std::size_t SIZE = 1<<16;

static void *randomize_buffer[SIZE];

void randomize_one_size_class(size_t, umpire::Allocator& alloc) {
  std::mt19937 gen(12345678);
  std::uniform_int_distribution<std::size_t> dist(64, 1024*1024);

  constexpr std::size_t count = SIZE;

  for (std::size_t repeat = 0; repeat < 10; repeat++) {
    for (std::size_t i = 0; i < count; i++) {
      randomize_buffer[i] = alloc.allocate(dist(gen));
    }
    std::random_shuffle(randomize_buffer, randomize_buffer + count);
    for (std::size_t i = 0; i < count; i++) {
      alloc.deallocate(randomize_buffer[i]);
    }
  }
}

int main() {
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc_new = rm.makeAllocator<umpire::strategy::Pool, false>(
      "POOL_NEW", rm.getAllocator("HOST"));

  auto alloc_old = rm.makeAllocator<umpire::strategy::DynamicPoolList, false>(
      "POOL_OLD", rm.getAllocator("HOST"));

  auto start = std::chrono::high_resolution_clock::now();
  randomize_one_size_class(8, alloc_new);
  auto end = std::chrono::high_resolution_clock::now();

  auto new_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "New took " << new_time << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  randomize_one_size_class(8, alloc_old);
  end = std::chrono::high_resolution_clock::now();
  auto old_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "Old took " << old_time << "ms" << std::endl;

  //int i = 4 << 10;
  //for (i = 16; i < 256; i += 16) {
  //  randomize_one_size_class(i, alloc);
  //}
  //for (; i < 512; i += 32) {
  //  randomize_one_size_class(i, alloc);
  //}
  //for (; i < 1024; i += 64) {
  //  randomize_one_size_class(i, alloc);
  //}
  //for (; i < (4 << 10); i += 128) {
  //  randomize_one_size_class(i, alloc);
  //}
  // for (; i < (32 << 10); i += 1024) {
  //   randomize_one_size_class(i, alloc);
  // }

  return 0;
}
