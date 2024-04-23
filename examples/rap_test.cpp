#include <iostream>
#include <stdio.h>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"

constexpr int BLOCK_SIZE = 16;
constexpr int NUM_THREADS = 64;

using clock_value_t = long long;

__device__ void sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < sleep_cycles);
}

__global__ void touch_data(double* data, int len)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < len) {
    data[id] = id;
  }
}

__global__ void do_sleep()
{
  //sleep - works still at 1000, so keeping it at 100k
  sleep(1000);
}

__global__ void check_data(double* data, int len)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  //Then error check that data[id] still == id
  if (id < len) {
    if (data[id] != id)
      data[id] = -1; 
  }
}

__global__ void touch_data_again(double* data, int len)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < len) {
    data[id] = 8.76543210;
  }
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("UM"));
  int NUM_BLOCKS = NUM_THREADS / BLOCK_SIZE;

  camp::resources::Cuda r1;
  camp::resources::Cuda r2;

  //allocate memory with s1 stream for a
  double* a = static_cast<double*>(pool.allocate(NUM_THREADS * sizeof(double), r1));

  //with stream s1, use memory in a in kernels
  touch_data<<<NUM_BLOCKS, BLOCK_SIZE, 0, r1.get_stream()>>>(a, NUM_THREADS);
  do_sleep<<<NUM_BLOCKS, BLOCK_SIZE, 0, r1.get_stream()>>>();
  check_data<<<NUM_BLOCKS, BLOCK_SIZE, 0, r1.get_stream()>>>(a, NUM_THREADS);

  //deallocate and reallocate a using different streams
  pool.deallocate(a, r1);
  a = static_cast<double*>(pool.allocate(NUM_THREADS * sizeof(double), r2));

  //with stream s2, use memory in reallocated a in kernel
  touch_data_again<<<NUM_BLOCKS, BLOCK_SIZE, 0, r2.get_stream()>>>(a, NUM_THREADS);

  //after this, all of this is just for checking/validation purposes
  double* b = static_cast<double*>(pool.allocate(NUM_THREADS * sizeof(double), r2));
  rm.copy(b, a);
  b = static_cast<double*>(rm.move(b, rm.getAllocator("HOST")));

  cudaDeviceSynchronize();

  std::cout << "Values are: " <<std::endl;
  for (int i = 0; i < NUM_THREADS; i++) {
    std::cout<< b[i] << " ";
  }
  for (int i = 0; i < NUM_THREADS; i++) {
    UMPIRE_ASSERT(b[i] != (-1) && "Error: incorrect value!");
  }
  std::cout << "Kernel succeeded! Expected result returned" << std::endl;

  //final deallocations
  pool.deallocate(a, r2);
  rm.deallocate(b);
  return 0;
}
