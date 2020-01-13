//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "benchmark/benchmark.h"

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
