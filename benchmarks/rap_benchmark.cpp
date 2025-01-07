#include <benchmark/benchmark.h>
#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/ResourceAwarePool.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include "camp/camp.hpp"
#include <vector>

#if defined(UMPIRE_ENABLE_CUDA)
using resource_type = camp::resources::Cuda;
#elif defined(UMPIRE_ENABLE_HIP)
using resource_type = camp::resources::Hip;
#endif

// Define the number of allocations and resources
const int N = 100; // Number of allocations
const int M = 4;   // Number of camp resources

// Define the size of each allocation
const size_t ALLOCATION_SIZE = 1024 * 1024; // 1 MB

// Dummy kernel function (assuming CUDA is available)
__global__ void dummy_kernel(double* data) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ALLOCATION_SIZE) {
    data[idx] = idx;
  }
}

// RAP Benchmark function
static void BM_ResourceAwarePoolAllocations(benchmark::State& state) {
  static int unique_id = 0;
  auto& rm = umpire::ResourceManager::getInstance();
  auto rap_pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool" + std::to_string(unique_id++), rm.getAllocator("UM"));

  // Create camp resources for device streams
  std::vector<resource_type> my_resources(M);

  for (auto _ : state) {
    std::vector<double*> my_allocations(N);

    // Perform N allocations across M resources
    for (int i = 0; i < N; ++i) {
      int ri = i % M;
      my_allocations[i] = static_cast<double*>(rap_pool.allocate(ALLOCATION_SIZE * sizeof(double), my_resources[ri]));

      // Optionally launch a dummy kernel to simulate work
      dummy_kernel<<<4096, 256, 0, my_resources[ri].get_stream()>>>(my_allocations[i]);
    }

    // Deallocate all allocations
    for (int i = 0; i < N; ++i) {
      rap_pool.deallocate(my_allocations[i]);
    }
  }
}

// QP Benchmark function
static void BM_QuickPoolAllocations(benchmark::State& state) {
  static int unique_id = 0;
  auto& rm = umpire::ResourceManager::getInstance();
  auto qp_pool = rm.makeAllocator<umpire::strategy::QuickPool>("qp-pool" + std::to_string(unique_id++), rm.getAllocator("UM"));

  // Create camp resources for device streams
  std::vector<resource_type> my_resources(M);

  for (auto _ : state) {
    std::vector<double*> my_allocations(N);

    // Perform N allocations across M resources
    for (int i = 0; i < N; ++i) {
      int ri = i % M;
      my_allocations[i] = static_cast<double*>(qp_pool.allocate(ALLOCATION_SIZE * sizeof(double)));

      // Optionally launch a dummy kernel to simulate work
      dummy_kernel<<<4096, 256, 0, my_resources[ri].get_stream()>>>(my_allocations[i]);
    }

    // Deallocate all allocations
    for (int i = 0; i < N; ++i) {
      qp_pool.deallocate(my_allocations[i]);
    }
  }
}

// Register the benchmark
BENCHMARK(BM_ResourceAwarePoolAllocations)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_QuickPoolAllocations)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
