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

static void *randomize_buffer[13<<20];

void randomize_one_size_class(size_t, umpire::Allocator& alloc) {
  std::mt19937 gen(12345678);
  std::uniform_int_distribution<std::size_t> dist(64, 1024*1024);

  constexpr std::size_t count = 13<<20;

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

  auto alloc = rm.makeAllocator<umpire::strategy::Pool, false>(
      "POOL", rm.getAllocator("HOST"));

  randomize_one_size_class(8, alloc);
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

  alloc.release();

  return 0;
}
