#include <stdio.h>
#include <math.h>
#include <iostream>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"

using namespace camp::resources;

constexpr int ITER = 5;
constexpr int NUM = 2048;
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
  // sleep - works still at 1000, so keeping it at 100k
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

void QuickPool_check(umpire::Allocator quick_pool)
{
  auto& rm = umpire::ResourceManager::getInstance();
  bool error{false};

  // Create hip streams
  hipStream_t s1, s2;
  hipStreamCreate(&s1); hipStreamCreate(&s2);

  double* a = static_cast<double*>(quick_pool.allocate(NUM * sizeof(double)));

  hipLaunchKernelGGL(touch_data, dim3(NUM_BLOCKS), dim3(NUM_PER_BLOCK), 0, s1, a);
  hipLaunchKernelGGL(do_sleep, dim3(NUM_BLOCKS), dim3(NUM_PER_BLOCK), 0, s1);
  hipLaunchKernelGGL(check_data, dim3(NUM_BLOCKS), dim3(NUM_PER_BLOCK), 0, s1, a);

  quick_pool.deallocate(a);
  a = static_cast<double*>(quick_pool.allocate(NUM * sizeof(double)));

  hipLaunchKernelGGL(touch_data_again, dim3(NUM_BLOCKS), dim3(NUM_PER_BLOCK), 0, s2, a);

  double* b = static_cast<double*>(quick_pool.allocate(NUM * sizeof(double)));
  rm.copy(b, a);
  b = static_cast<double*>(rm.move(b, rm.getAllocator("HOST")));

  for (int i = 0; i < NUM; i++) {
    if(b[i] == (-1)) {
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
  rm.deallocate(b);
  hipStreamDestroy(s1); hipStreamDestroy(s2);
}

void ResourceAwarePool_check(umpire::Allocator rap_pool)
{
  // Create hip resources
  Hip d1, d2;
  Resource r1{d1}, r2{d2};

  // ResourceAwarePool checks
  auto& rm = umpire::ResourceManager::getInstance();
  bool error{false};

  for(int i = 0; i < ITER; i++) {
    double* a = static_cast<double*>(rap_pool.allocate(r1, NUM * sizeof(double)));

    hipLaunchKernelGGL(touch_data, dim3(NUM_BLOCKS), dim3(NUM_PER_BLOCK), 0, d1.get_stream(), a);
    hipLaunchKernelGGL(do_sleep, dim3(NUM_BLOCKS), dim3(NUM_PER_BLOCK), 0, d1.get_stream());
    hipLaunchKernelGGL(check_data, dim3(NUM_BLOCKS), dim3(NUM_PER_BLOCK), 0, d1.get_stream(), a);

    rap_pool.deallocate(r1, a);
    a = static_cast<double*>(rap_pool.allocate(r2, NUM * sizeof(double)));

    hipLaunchKernelGGL(touch_data_again, dim3(NUM_BLOCKS), dim3(NUM_PER_BLOCK), 0, d2.get_stream(), a);

    double* b = static_cast<double*>(rap_pool.allocate(r2, NUM * sizeof(double)));
    rm.copy(b, a);
    b = static_cast<double*>(rm.move(b, rm.getAllocator("HOST")));

    for (int i = 0; i < NUM; i++) {
      if(b[i] == (-1)) {
        error = true;
        break;
      }
    }

    if (error) {
      std::cout << "Errors Found!" << std::endl;
    } else {
      std::cout << "Kernel succeeded! Expected result returned" << std::endl;
    }

    rap_pool.deallocate(r2, a);
    rm.deallocate(b);
    error = false; // reset to find any new errors in next iter
  }
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto quick_pool = rm.makeAllocator<umpire::strategy::QuickPool>("quick-pool", rm.getAllocator("UM"));
  auto rap_pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("UM"));

  std::cout<<"Checking QuickPool ...."<<std::endl;
  QuickPool_check(quick_pool);

  std::cout<<"Checking ResourceAwarePool ...."<<std::endl;
  ResourceAwarePool_check(rap_pool);

  std::cout<<"Done!"<<std::endl;
  return 0;
}
  
