#include "benchmark/benchmark_api.h"

#include "umpire/alloc/MallocAllocator.hpp"
#include "umpire/alloc/CudaMallocAllocator.hpp"
#include "umpire/alloc/Pool.hpp"

#include <climits>

static void benchmark_malloc(benchmark::State& state) {
  auto allocator = umpire::alloc::MallocAllocator();

  while (state.KeepRunning()) {
    void* ptr = allocator.malloc(state.range_x());
    allocator.free(ptr);
  }
}

static void benchmark_pool_malloc(benchmark::State& state) {
  umpire::alloc::Pool<> pool;

  while (state.KeepRunning()) {
    void* ptr = pool.allocate(state.range_x());
    pool.free(ptr);
  }
}

static void benchmark_pool_device_malloc(benchmark::State& state) {
  umpire::alloc::Pool<umpire::alloc::CudaMallocAllocator> pool;

  while (state.KeepRunning()) {
    void* ptr = pool.allocate(state.range_x());
    pool.free(ptr);
  }
}

BENCHMARK(benchmark_malloc)->Range(1, INT_MAX);
BENCHMARK(benchmark_pool_malloc)->Range(1, INT_MAX);
BENCHMARK(benchmark_pool_device_malloc)->Range(1, INT_MAX);

BENCHMARK_MAIN();
