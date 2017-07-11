#include "benchmark/benchmark_api.h"

#include "umpire/alloc/CudaMallocAllocator.hpp"

static void benchmark_malloc_free(benchmark::State& state) {
  auto allocator = umpire::alloc::CudaMallocAllocator();

  while (state.KeepRunning()) {
    void* ptr = allocator.allocate(state.range_x());
    allocator.free(ptr);
  }
}

BENCHMARK(benchmark_malloc_free)->Range(8, 8<<24);

BENCHMARK_MAIN();
