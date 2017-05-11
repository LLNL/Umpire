#include "benchmark/benchmark_api.h"

#include "umpire/alloc/MallocAllocator.hpp"
#include "umpire/alloc/Pool.hpp"

static void benchmark_malloc(benchmark::State& state) {
  auto allocator = umpire::alloc::MallocAllocator();

  while (state.KeepRunning()) {
    void* ptr = allocator.allocate(state.range_x());
    allocator.free(ptr);
  }
}

static void benchmark_pool_malloc(benchmark::State& state) {
  auto allocator = umpire::alloc::MallocAllocator();

  while (state.KeepRunning()) {
    void* ptr = allocator.allocate(state.range_x());
    allocator.free(ptr);
  }
}
