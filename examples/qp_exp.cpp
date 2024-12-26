#include <stdio.h>
#include <math.h>
#include <iostream>

#include <thread>
#include <chrono>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
using resource_type = camp::resources::Cuda;
#elif defined(UMPIRE_ENABLE_HIP)
using resource_type = camp::resources::Hip;
#endif

constexpr int NUM = 1 << 18;
const int NUM_PER_BLOCK = 256;
const int NUM_BLOCKS = NUM/NUM_PER_BLOCK;

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

__global__ void do_sleep()
{
  // Sleep in kernel in order to replicate data race
  sleep(100000000);
}

__global__ void touch_data(double* data)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < NUM) {
    data[id] = id;
  }
}

__global__ void check_data(double* data)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  //Then error check that data[id] still == id
  if (id < NUM) {
    if (data[id] != id)
      data[id] = -1; 
  }
}

__global__ void touch_data_again(double* data)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < NUM) {
    data[id] = 8.76543210;
  }
}

int main(int, char**)
{
  auto start_total = std::chrono::high_resolution_clock::now();

  auto& rm = umpire::ResourceManager::getInstance();
  auto quick_pool = rm.makeAllocator<umpire::strategy::QuickPool>("quick-pool", rm.getAllocator("UM"));

  std::cout<<"Checking QuickPool ...."<<std::endl;
  bool error{false};

  // Create camp resources for device streams
  resource_type s1, s2;

  double* a = static_cast<double*>(quick_pool.allocate(NUM * sizeof(double)));

  touch_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, s1.get_stream()>>>(a);
  do_sleep<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, s1.get_stream()>>>();
  check_data<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, s1.get_stream()>>>(a);

  quick_pool.deallocate(a);

  auto start_do_sync = std::chrono::high_resolution_clock::now();
  s1.get_event().wait();  
  auto end_do_sync = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_do_sync = end_do_sync - start_do_sync;
  std::cout << "wait call time: " << duration_do_sync.count() << " seconds" << std::endl;

  a = static_cast<double*>(quick_pool.allocate(NUM * sizeof(double)));

  touch_data_again<<<NUM_BLOCKS, NUM_PER_BLOCK, 0, s2.get_stream()>>>(a);

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (int i = 0; i < NUM; i++) {
    if(a[i] == (-1)) {
      error = true;
      break;
    }
  }

  if (error) {
    std::cout << "Errors Found!" << std::endl;
  } else {
    std::cout << "Kernel succeeded! Expected result returned" << std::endl;
  }

  quick_pool.deallocate(a);

  auto end_total = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_total = end_total - start_total;
  std::cout << "Total execution time for QP: " << duration_total.count() << " seconds" << std::endl;

  return 0;
}
  
