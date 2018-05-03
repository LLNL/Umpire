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

static const size_t max_allocations = 100000;

static void benchmark_allocate(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);
  void** allocations = new void*[max_allocations];

  auto size = state.range(0);

  size_t i = 0;
  while (state.KeepRunning()) {
    if ( i == max_allocations ) {
      state.PauseTiming();
      for (size_t j = 0; j < max_allocations; j++)
        allocator.deallocate(allocations[j]);
      i = 0;
      state.ResumeTiming();
    }
    allocations[i++] = allocator.allocate(size);
  }

  for (size_t j = 0; j < i; j++)
    allocator.deallocate(allocations[j]);

  delete[] allocations;
}

static void benchmark_deallocate(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);

  void** allocations = new void*[max_allocations];
  auto size = state.range(0);

  size_t i = 0;
  while (state.KeepRunning()) {
    if ( i == 0 || i == max_allocations ) {
      state.PauseTiming();
      for (size_t j = 0; j < max_allocations; j++)
        allocations[j] = allocator.allocate(size);
      i = 0;
      state.ResumeTiming();
    }
    allocator.deallocate(allocations[i++]);
  }

  for (size_t j = i; j < max_allocations; j++)
    allocator.deallocate(allocations[j]);
  delete[] allocations;
}

static void benchmark_malloc(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);
  void** allocations = new void*[max_allocations];

  auto size = state.range(0);

  size_t i = 0;
  while (state.KeepRunning()) {
    if ( i == max_allocations ) {
      state.PauseTiming();
      for (size_t j = 0; j < max_allocations; j++)
        free(allocations[j]);
      i = 0;
      state.ResumeTiming();
    }

    allocations[i++] = malloc(size);
  }

  for (size_t j = 0; j < i; j++)
    free(allocations[j]);

  delete[] allocations;
}

static void benchmark_free(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);

  void** allocations = new void*[max_allocations];
  auto size = state.range(0);

  size_t i = 0;
  while (state.KeepRunning()) {
    if ( i == 0 || i == max_allocations ) {
      state.PauseTiming();
      for (size_t j = 0; j < max_allocations; j++)
        allocations[j] = malloc(size);
      i = 0;
      state.ResumeTiming();
    }
    free(allocations[i++]);
  }

  for (size_t j = i; j < max_allocations; j++)
    free(allocations[j]);
  delete[] allocations;
}

#define RUNALL
#ifdef RUNALL
BENCHMARK_CAPTURE(benchmark_allocate,   host, std::string("HOST"))->Range(4, 1024);
BENCHMARK_CAPTURE(benchmark_malloc,  host, std::string("HOST"))->Range(4, 1024);
BENCHMARK_CAPTURE(benchmark_deallocate, host, std::string("HOST"))->Range(4, 1024);
BENCHMARK_CAPTURE(benchmark_free,       host, std::string("HOST"))->Range(4, 1024);
#endif

#if defined(UMPIRE_ENABLE_CUDA)
BENCHMARK_CAPTURE(benchmark_allocate, um, std::string("UM"))->Range(4, 1024);
BENCHMARK_CAPTURE(benchmark_deallocate, um, std::string("UM"))->Range(4, 1024);

BENCHMARK_CAPTURE(benchmark_allocate, device, std::string("DEVICE"))->Range(4, 1024);
BENCHMARK_CAPTURE(benchmark_deallocate, device, std::string("DEVICE"))->Range(4, 1024);
#endif

BENCHMARK_MAIN();
