#include <iostream>

#include "benchmark/benchmark_api.h"

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"


static void benchmark_allocate(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);
  void** allocations = new void*[state.max_iterations];

  auto size = state.range(0);

  size_t i = 0;
  while (state.KeepRunning()) {
    allocations[i++] = allocator.allocate(size);
  }
}

static void benchmark_deallocate(benchmark::State& state, std::string name) {
  auto allocator = umpire::ResourceManager::getInstance().getAllocator(name);

  void** allocations = new void*[state.max_iterations];
  auto size = state.range(0);

  for (size_t i = 0; i < state.max_iterations; i++) {
    allocations[i] = allocator.allocate(size);
  }

  size_t i = 0;
  while (state.KeepRunning()) {
    allocator.deallocate(allocations[i++]);
  }
}

BENCHMARK_CAPTURE(benchmark_allocate, host, std::string("HOST"))->Range(4, 4096);

BENCHMARK_CAPTURE(benchmark_deallocate, host, std::string("HOST"))->Range(4, 4096);

#if defined(ENABLE_CUDA)
BENCHMARK_CAPTURE(benchmark_allocate, um, std::string("UM"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_deallocate, um, std::string("UM"))->Range(4, 4096);

BENCHMARK_CAPTURE(benchmark_allocate, device, std::string("DEVICE"))->Range(4, 4096);
BENCHMARK_CAPTURE(benchmark_deallocate, device, std::string("DEVICE"))->Range(4, 4096);
#endif

BENCHMARK_MAIN();
