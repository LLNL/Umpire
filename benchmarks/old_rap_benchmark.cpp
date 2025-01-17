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

constexpr int NUM_ALLOC = 100;
const int NUM_RES = 4;
constexpr int SIZE = 1 << 21;
const int NUM_PER_BLOCK = 256;
const int NUM_BLOCKS = SIZE/NUM_PER_BLOCK;

__global__ void touch_data(double* data, int i)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < SIZE) {
    data[id] = id + i;
  }
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto rap_pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("UM"));
  auto qp_pool = rm.makeAllocator<umpire::strategy::QuickPool>("qp-pool", rm.getAllocator("UM"));
  double* a;

  // Create camp resources for device streams
  std::vector<resource_type> resources(NUM_RES);

  std::cout<<"Timing " << NUM_ALLOC << " allocations and " << NUM_RES << " resources with the QuickPool ...."<<std::endl;
  std::chrono::duration<double> duration_total[NUM_ALLOC];
  std::chrono::duration<double> this_duration_total[NUM_ALLOC];
  std::chrono::time_point<std::chrono::high_resolution_clock> my_start, my_end;

  for (int r = 0; r < NUM_RES; r++) {
    for (int i = 0; i < NUM_ALLOC; i++) {
      auto start_total = std::chrono::high_resolution_clock::now();
      a = static_cast<double*>(qp_pool.allocate(SIZE * sizeof(double)));
      touch_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, resources[r].get_stream()>>>(a, i);
      qp_pool.deallocate(a);
      auto end_total = std::chrono::high_resolution_clock::now();
      duration_total[i] = end_total - start_total;
    }

    // Calculate average, max, and min durations
    double total_duration = 0.0;
    for (const auto& duration : duration_total) {
      total_duration += duration.count();
    }
    double average_duration = total_duration / NUM_ALLOC;

    auto min_duration = *std::min_element(duration_total, duration_total + NUM_ALLOC);
    auto max_duration = *std::max_element(duration_total, duration_total + NUM_ALLOC);

    std::cout << "Resource " << r << " statistics:" << std::endl;
    std::cout << "Average execution time: " << (average_duration * 1000.0) << " milliseconds" << std::endl;
    std::cout << "Minimum execution time: " << (min_duration.count() * 1000.0) << " milliseconds" << std::endl;
    std::cout << "Maximum execution time: " << (max_duration.count() * 1000.0) << " milliseconds" << std::endl;
  }

  std::cout<< std::endl;
  std::cout<<"Timing " << NUM_ALLOC << " allocations ACROSS " << NUM_RES << " resources with the QuickPool ...."<<std::endl;

  {
    for (int i = 0; i < NUM_ALLOC; i++) {
      int ri = i % NUM_RES;

      my_start = std::chrono::high_resolution_clock::now();
      a = static_cast<double*>(qp_pool.allocate(SIZE * sizeof(double)));
      touch_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, resources[ri].get_stream()>>>(a, i);
      qp_pool.deallocate(a);
      my_end = std::chrono::high_resolution_clock::now();
      this_duration_total[i] = my_end - my_start;
    }

    // Calculate average, max, and min durations
    double total_duration = 0.0;
    for (const auto& duration : this_duration_total) {
      total_duration += duration.count();
    }
    double average_duration = total_duration / NUM_ALLOC;

    auto min_duration = *std::min_element(duration_total, duration_total + NUM_ALLOC);
    auto max_duration = *std::max_element(duration_total, duration_total + NUM_ALLOC);

    std::cout << "Average execution time: " << (average_duration * 1000.0) << " milliseconds" << std::endl;
    std::cout << "Minimum execution time: " << (min_duration.count() * 1000.0) << " milliseconds" << std::endl;
    std::cout << "Maximum execution time: " << (max_duration.count() * 1000.0) << " milliseconds" << std::endl;
  }

  std::cout<< std::endl;
  std::cout<<"Timing " << NUM_ALLOC << " allocations and " << NUM_RES << " resources with the ResourceAwarePool ...."<<std::endl;

  for (int r = 0; r < NUM_RES; r++) {
    for (int i = 0; i < NUM_ALLOC; i++) {
      auto start_total = std::chrono::high_resolution_clock::now();
      a = static_cast<double*>(rap_pool.allocate(SIZE * sizeof(double), resources[r]));
      touch_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, resources[r].get_stream()>>>(a, i);
      rap_pool.deallocate(a);
      auto end_total = std::chrono::high_resolution_clock::now();
      duration_total[i] = end_total - start_total;
    }

    // Calculate average, max, and min durations
    double total_duration = 0.0;
    for (const auto& duration : duration_total) {
      total_duration += duration.count();
    }
    double average_duration = total_duration / NUM_ALLOC;

    auto min_duration = *std::min_element(duration_total, duration_total + NUM_ALLOC);
    auto max_duration = *std::max_element(duration_total, duration_total + NUM_ALLOC);

    std::cout << "Resource " << r << " statistics:" << std::endl;
    std::cout << "Average execution time: " << (average_duration * 1000.0) << " milliseconds" << std::endl;
    std::cout << "Minimum execution time: " << (min_duration.count() * 1000.0) << " milliseconds" << std::endl;
    std::cout << "Maximum execution time: " << (max_duration.count() * 1000.0) << " milliseconds" << std::endl;
  }

  std::cout<< std::endl;
  std::cout<<"Timing " << NUM_ALLOC << " allocations ACROSS " << NUM_RES << " resources with the ResourceAwarePool ...."<<std::endl;

  {
    for (int i = 0; i < NUM_ALLOC; i++) {
      int ri = i % NUM_RES;

      my_start = std::chrono::high_resolution_clock::now();
      a = static_cast<double*>(rap_pool.allocate(SIZE * sizeof(double), resources[ri]));
      touch_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, resources[ri].get_stream()>>>(a, i);
      rap_pool.deallocate(a);
      my_end = std::chrono::high_resolution_clock::now();
      this_duration_total[i] = my_end - my_start;
    }

    // Calculate average, max, and min durations
    double total_duration = 0.0;
    for (const auto& duration : this_duration_total) {
      total_duration += duration.count();
    }
    double average_duration = total_duration / NUM_ALLOC;

    auto min_duration = *std::min_element(duration_total, duration_total + NUM_ALLOC);
    auto max_duration = *std::max_element(duration_total, duration_total + NUM_ALLOC);

    std::cout << "Average execution time: " << (average_duration * 1000.0) << " milliseconds" << std::endl;
    std::cout << "Minimum execution time: " << (min_duration.count() * 1000.0) << " milliseconds" << std::endl;
    std::cout << "Maximum execution time: " << (max_duration.count() * 1000.0) << " milliseconds" << std::endl;
  }

  return 0;
}
  
