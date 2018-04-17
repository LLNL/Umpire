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

#include "umpire/util/Macros.hpp"

static void benchmark_DebugLogger(benchmark::State& state) {
  while (state.KeepRunning()) {
    UMPIRE_LOG(Debug, "(" << 22 << ")");
  }
}
//
// Register the function as a benchmark
BENCHMARK(benchmark_DebugLogger);

BENCHMARK_MAIN();
