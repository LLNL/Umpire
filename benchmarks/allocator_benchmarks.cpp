#include <iostream>

#include "benchmark/benchmark_api.h"

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"


static void benchmark_allocator(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);
  void** allocations = new void*[state.max_iterations];

  auto size = state.range(0);

  size_t i = 0;
  while (state.KeepRunning()) {
    allocations[i++] = allocator.allocate(size);
  }

  for (i = 0; i < state.max_iterations; i++) {
    allocator.deallocate(allocations[i]);
  }
}

BENCHMARK_CAPTURE(benchmark_allocator, host, std::string("HOST"))->Range(4, 4096);

#if defined(ENABLE_CUDA)
BENCHMARK_CAPTURE(benchmark_allocator, um, std::string("UM"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_allocator, device, std::string("DEVICE"))->Range(4, 4096);
#endif

BENCHMARK_MAIN();
