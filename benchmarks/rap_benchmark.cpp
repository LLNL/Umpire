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
constexpr int SIZE = 1 << 18;
//const int NUM_PER_BLOCK = 256;
//const int NUM_BLOCKS = SIZE/NUM_PER_BLOCK;

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
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("UM"));
  //auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("qp-pool", rm.getAllocator("UM"));

  double* a[NUM_ALLOC];

  // Create camp resources for device streams
  resource_type d1, d2;

  //Fill the pending list
  for( int i = 0; i < NUM_ALLOC; i ++) {
    a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double), d1));
  }
  for( int i = 0; i < NUM_ALLOC; i ++) {
    pool.deallocate(a[i]);
  }

  //Reallocate with the same resource
  {
    auto start_total = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i ++) {
      a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double), d1));
    }
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1 = end_total - start_total;
    std::cout << "Total execution time for reallocating with the SAME resource: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  }

  for( int i = 0; i < NUM_ALLOC; i ++) {
    pool.deallocate(a[i]);
  }
  
  //Fill the pending list
  for( int i = 0; i < NUM_ALLOC; i ++) {
    a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double), d1));
  }
  for( int i = 0; i < NUM_ALLOC; i ++) {
    pool.deallocate(a[i]);
  }

  //Reallocate with different resource
  {
    auto start_total = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < NUM_ALLOC; i++) {
      a[i] = static_cast<double*>(pool.allocate(SIZE * sizeof(double), d2));
    }
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total1 = end_total - start_total;
    std::cout << "Total execution time for reallocating with a DIFFERENT resource: " << (duration_total1.count() / NUM_ALLOC * 1000.0) << " milliseconds" << std::endl;
  }

  for( int i = 0; i < NUM_ALLOC; i ++) {
    pool.deallocate(a[i]);
  }

  return 0;
}
  
