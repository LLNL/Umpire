#include "umpire/CudaMallocAllocator.hpp"

#include "benchmark/benchmark_api.h"

static void benchmark_malloc_free(benchmark::State& state) {
  auto allocator = umpire::CudaMallocAllocator();

  while (state.KeepRunning()) {
    void* ptr = allocator.alloc(state.range_x());
    allocator.free(ptr);
  }
}

BENCHMARK(benchmark_malloc)->Range(8, 8<<24);

BENCHMARK_MAIN();
