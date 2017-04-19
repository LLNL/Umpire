#include "umpire/Umpire.hpp"

#include "umpire/MallocAllocator.hpp"

#include "benchmark/benchmark_api.h"

static void benchmark_malloc(benchmark::State& state) {
  auto allocator = umpire::MallocAllocator();

  while (state.KeepRunning()) {
    void* ptr = allocator.allocate(state.range_x());
    allocator.free(ptr);
  }
}

static void benchmark_malloc_ui(benchmark::State& state) {
  while (state.KeepRunning()) {
    void* ptr = umpire::malloc(state.range_x());
    umpire::free(ptr);
  }
}

BENCHMARK(benchmark_malloc)->Range(8, 8<<24);
BENCHMARK(benchmark_malloc_ui)->Range(8, 8<<24);

BENCHMARK_MAIN();
