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
#include <string>
#include <sstream>
#include <cassert>
#include <memory>

#include "benchmark/benchmark.h"

#include "umpire/config.hpp"

#include "umpire/strategy/mixins/Inspector.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

static void benchmark_inspector(benchmark::State& state) {
  auto inspector = umpire::strategy::mixins::Inspector();

  double* data = new double[256];

  void* ptr = static_cast<void*>(data);
  size_t length = 256*sizeof(double);
  std::shared_ptr<umpire::strategy::AllocationStrategy> strat(nullptr);

  while (state.KeepRunning()) {
    inspector.registerAllocation(
        ptr,
        length,
        strat);
  }

  delete[] data;
}

BENCHMARK(benchmark_inspector);

BENCHMARK_MAIN();
