#include <stdio.h>
#include <math.h>
#include <iostream>

#include <thread>
#include <chrono>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"
#include "umpire/strategy/QuickPool.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
using resource_type = camp::resources::Cuda;
#elif defined(UMPIRE_ENABLE_HIP)
using resource_type = camp::resources::Hip;
#endif

constexpr int NUM_ALLOC = 10;
const int NUM_RES = 4;
constexpr int SIZE = 1 << 21;
const int NUM_PER_BLOCK = 256;
const int NUM_BLOCKS = SIZE/NUM_PER_BLOCK;

using clock_value_t = long long;

__device__ clock_value_t my_clock()
{
  return clock64();
}

__device__ void sleep(clock_value_t sleep_cycles)
{
  clock_value_t start = my_clock();
  clock_value_t cycles_elapsed;
  do {
    cycles_elapsed = my_clock() - start;
  } while (cycles_elapsed < sleep_cycles);
}

__global__ void touch_data(double* data, int i)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id % 2 == 0) {
    data[id] = id + i;
    sleep(100000);
  } else {
    data[id] = id * i;
    sleep(10000);
  }
  if (id % 2 == 0) {
    sleep(100000000 + i);
  } else {
    sleep(100000 + (2*i));
  }
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto rap_pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("UM"));
  auto qp_pool = rm.makeAllocator<umpire::strategy::QuickPool>("qp-pool", rm.getAllocator("UM"));
  double* a1[NUM_ALLOC];

  // Create camp resources for device streams
  std::vector<resource_type> resources(NUM_RES);

  std::cout<<"Timing " << NUM_ALLOC << " allocations, run a long kernel, dealloc everything, and then allocate again with the QuickPool ...."<<std::endl;

  for (int r = 0; r < NUM_RES-1; r++) {
    auto start_total = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ALLOC; i++) {
      a1[i] = static_cast<double*>(qp_pool.allocate(SIZE * sizeof(double)));
      touch_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, resources[r].get_stream()>>>(a1[i], i);
    }
    for (int i = 0; i < NUM_ALLOC; i++) {
      qp_pool.deallocate(a1[i]);
    }
    for (int i = 0; i < NUM_ALLOC; i++) {
      a1[i] = static_cast<double*>(qp_pool.allocate(SIZE * sizeof(double)));
      touch_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, resources[r+1].get_stream()>>>(a1[i], i);
    }
    for (int i = 0; i < NUM_ALLOC; i++) {
      qp_pool.deallocate(a1[i]);
    }
    resources[r].get_event().wait();
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1 = end_total - start_total;

    std::cout << "Execution time: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  }

  std::cout<< std::endl;
  std::cout<<"Timing " << NUM_ALLOC << " allocations, run a long kernel, dealloc everything, and then allocate again with " << NUM_RES << "resources and the ResourceAwarePool ...."<<std::endl;

  for (int r = 0; r < NUM_RES-1; r++) {
    auto start_total = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ALLOC; i++) {
      a1[i] = static_cast<double*>(rap_pool.allocate(SIZE * sizeof(double), resources[r]));
      touch_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, resources[r].get_stream()>>>(a1[i], i);
    }
    for (int i = 0; i < NUM_ALLOC; i++) {
      rap_pool.deallocate(a1[i]);
    }
    for (int i = 0; i < NUM_ALLOC; i++) {
      a1[i] = static_cast<double*>(rap_pool.allocate(SIZE * sizeof(double), resources[r+1]));
      touch_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, resources[r+1].get_stream()>>>(a1[i], i);
    }
    for (int i = 0; i < NUM_ALLOC; i++) {
      rap_pool.deallocate(a1[i]);
    }
    resources[r].get_event().wait();
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1 = end_total - start_total;

    std::cout << "Execution time: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  }
    
  std::cout << "Done. " << std::endl;
  return 0;
}
