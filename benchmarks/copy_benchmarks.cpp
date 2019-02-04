//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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


static void benchmark_copy(benchmark::State& state, std::string src, std::string dest) {
  auto& rm = umpire::ResourceManager::getInstance();

  auto source_allocator = rm.getAllocator(src);
  auto dest_allocator = rm.getAllocator(dest);

  auto size = state.range(0);

  void* src_ptr = source_allocator.allocate(size);
  void* dest_ptr = dest_allocator.allocate(size);

  while (state.KeepRunning()) {
    rm.copy(src_ptr, dest_ptr);
  }

  source_allocator.deallocate(src_ptr);
  dest_allocator.deallocate(dest_ptr);
}

BENCHMARK_CAPTURE(benchmark_copy, host_host, std::string("HOST"), std::string("HOST"))->Range(4, 4096);

#if defined(UMPIRE_ENABLE_CUDA)
BENCHMARK_CAPTURE(benchmark_copy, host_device, std::string("HOST"), std::string("DEVICE"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_copy, device_host, std::string("DEVICE"), std::string("HOST"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_copy, device_device, std::string("DEVICE"), std::string("DEVICE"))->Range(4, 4096);

BENCHMARK_CAPTURE(benchmark_copy, host_device, std::string("HOST"), std::string("UM"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_copy, device_host, std::string("UM"), std::string("HOST"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_copy, device_device, std::string("UM"), std::string("UM"))->Range(4, 4096);

BENCHMARK_CAPTURE(benchmark_copy, device_device, std::string("DEVICE"), std::string("UM"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_copy, device_device, std::string("UM"), std::string("DEVICE"))->Range(4, 4096);
#endif

BENCHMARK_MAIN();
