//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "benchmark/benchmark_api.h"

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"


static void benchmark_allocate(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);
  void** allocations = new void*[state.max_iterations];

  auto size = state.range(0);

  size_t i = 0;
  while (state.KeepRunning()) {
    allocations[i++] = allocator.allocate(size);
  }

  for (size_t j = 0; j < i; j++)
    allocator.deallocate(allocations[j]);

  delete[] allocations;
}

static void benchmark_deallocate(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);

  void** allocations = new void*[state.max_iterations];
  auto size = state.range(0);

  for (size_t i = 0; i < state.max_iterations; i++) {
    allocations[i] = allocator.allocate(size);
  }

  size_t i = 0;
  while (state.KeepRunning()) {
    allocator.deallocate(allocations[i++]);
  }

  for (size_t j = i; j < state.max_iterations; j++)
    allocator.deallocate(allocations[j]);
  delete[] allocations;
}

static void benchmark_mallocate(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);
  void** allocations = new void*[state.max_iterations];

  auto size = state.range(0);

  size_t i = 0;
  while (state.KeepRunning()) {
    allocations[i++] = malloc(size);
  }

  for (size_t j = 0; j < i; j++)
    free(allocations[j]);

  delete[] allocations;
}

static void benchmark_free(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);

  void** allocations = new void*[state.max_iterations];
  auto size = state.range(0);

  for (size_t i = 0; i < state.max_iterations; i++) {
    allocations[i] = malloc(size);
  }

  size_t i = 0;
  while (state.KeepRunning()) {
    free(allocations[i++]);
  }

  for (size_t j = i; j < state.max_iterations; j++)
    free(allocations[j]);
  delete[] allocations;
}

BENCHMARK_CAPTURE(benchmark_allocate,   host, std::string("HOST"))->Range(4, 4);
BENCHMARK_CAPTURE(benchmark_mallocate,  host, std::string("HOST"))->Range(4, 4);
BENCHMARK_CAPTURE(benchmark_deallocate, host, std::string("HOST"))->Range(4, 4);
BENCHMARK_CAPTURE(benchmark_free,       host, std::string("HOST"))->Range(4, 4);

BENCHMARK_CAPTURE(benchmark_allocate,   host, std::string("HOST"))->Range(8, 8);
BENCHMARK_CAPTURE(benchmark_mallocate,  host, std::string("HOST"))->Range(8, 8);
BENCHMARK_CAPTURE(benchmark_deallocate, host, std::string("HOST"))->Range(8, 8);
BENCHMARK_CAPTURE(benchmark_free,       host, std::string("HOST"))->Range(8, 8);

BENCHMARK_CAPTURE(benchmark_allocate,   host, std::string("HOST"))->Range(16, 16);
BENCHMARK_CAPTURE(benchmark_mallocate,  host, std::string("HOST"))->Range(16, 16);
BENCHMARK_CAPTURE(benchmark_deallocate, host, std::string("HOST"))->Range(16, 16);
BENCHMARK_CAPTURE(benchmark_free,       host, std::string("HOST"))->Range(16, 16);

BENCHMARK_CAPTURE(benchmark_allocate,   host, std::string("HOST"))->Range(32, 32);
BENCHMARK_CAPTURE(benchmark_mallocate,  host, std::string("HOST"))->Range(32, 32);
BENCHMARK_CAPTURE(benchmark_deallocate, host, std::string("HOST"))->Range(32, 32);
BENCHMARK_CAPTURE(benchmark_free,       host, std::string("HOST"))->Range(32, 32);

BENCHMARK_CAPTURE(benchmark_allocate,   host, std::string("HOST"))->Range(64, 64);
BENCHMARK_CAPTURE(benchmark_mallocate,  host, std::string("HOST"))->Range(64, 64);
BENCHMARK_CAPTURE(benchmark_deallocate, host, std::string("HOST"))->Range(64, 64);
BENCHMARK_CAPTURE(benchmark_free,       host, std::string("HOST"))->Range(64, 64);

BENCHMARK_CAPTURE(benchmark_allocate,   host, std::string("HOST"))->Range(512, 512);
BENCHMARK_CAPTURE(benchmark_mallocate,  host, std::string("HOST"))->Range(512, 512);
BENCHMARK_CAPTURE(benchmark_deallocate, host, std::string("HOST"))->Range(512, 512);
BENCHMARK_CAPTURE(benchmark_free,       host, std::string("HOST"))->Range(512, 512);

BENCHMARK_CAPTURE(benchmark_allocate,   host, std::string("HOST"))->Range(4096, 4096);
BENCHMARK_CAPTURE(benchmark_mallocate,  host, std::string("HOST"))->Range(4096, 4096);
BENCHMARK_CAPTURE(benchmark_deallocate, host, std::string("HOST"))->Range(4096, 4096);
BENCHMARK_CAPTURE(benchmark_free,       host, std::string("HOST"))->Range(4096, 4096);

#if defined(UMPIRE_ENABLE_CUDA)
BENCHMARK_CAPTURE(benchmark_allocate, um, std::string("UM"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_deallocate, um, std::string("UM"))->Range(4, 4096);

BENCHMARK_CAPTURE(benchmark_allocate, device, std::string("DEVICE"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_deallocate, device, std::string("DEVICE"))->Range(4, 4096);
#endif

BENCHMARK_MAIN();
