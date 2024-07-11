#include <iostream>
#include <stdio.h>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"

constexpr int BLOCK_SIZE = 16;
constexpr int NUM_THREADS = 64;

using clock_value_t = long long;
using namespace camp::resources;

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
  sleep(1000000);
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

  //Create camp resources
#if defined(UMPIRE_ENABLE_CUDA)
  Cuda d1, d2, d3;
#elif defined(UMPIRE_ENABLE_HIP)
  Hip d1, d2, d3;
#else
  Host d1, d2, d3;
#endif
  Resource r1{d1}, r2{d2}, r3{d3};

  //allocate memory in the pool with r1
  double* a = static_cast<double*>(pool.allocate(r1, NUM_THREADS * sizeof(double)));
  std::cout << "HERE1" << std::endl;

  //launch kernels on r1's stream
  touch_data<<<NUM_BLOCKS, BLOCK_SIZE, 0, d1.get_stream()>>>(a, NUM_THREADS);
  do_sleep<<<NUM_BLOCKS, BLOCK_SIZE, 0, d1.get_stream()>>>();
  check_data<<<NUM_BLOCKS, BLOCK_SIZE, 0, d1.get_stream()>>>(a, NUM_THREADS);
  std::cout << "HERE2" << std::endl;

  //deallocate memory with r1 and reallocate using a different stream r2
  pool.deallocate(r1, a);
  std::cout << "HERE3" << std::endl;
  a = static_cast<double*>(pool.allocate(r2, NUM_THREADS * sizeof(double)));
  std::cout << "HERE4" << std::endl;

  //launch kernel with r2's stream using newly reallocated 'a'
  touch_data_again<<<NUM_BLOCKS, BLOCK_SIZE, 0, d2.get_stream()>>>(a, NUM_THREADS);
  std::cout << "HERE5" << std::endl;

  //bring final data from 'a' back to host var 'b'
  double* b = static_cast<double*>(pool.allocate(r2, NUM_THREADS * sizeof(double)));
  std::cout << "HERE6" << std::endl;
  rm.copy(b, a);
  b = static_cast<double*>(rm.move(b, rm.getAllocator("HOST")));
  std::cout << "HERE7" << std::endl;

  //For validation/error checking below, synchronize host and device
#if defined(UMPIRE_ENABLE_CUDA)
  cudaDeviceSynchronize();
#elif defined(UMPIRE_ENABLE_HIP)
  hipDeviceSynchronize();
#endif

  std::cout << "HERE8" << std::endl;
  //Error check and validation
  std::cout << "Values are: " <<std::endl;
  for (int i = 0; i < NUM_THREADS; i++) {
    std::cout<< b[i] << " ";
  }
  for (int i = 0; i < NUM_THREADS; i++) {
    UMPIRE_ASSERT(b[i] != (-1) && "Error: incorrect value!");
  }
  std::cout << "Kernel succeeded! Expected result returned" << std::endl;

  //deallocate and clean up
  pool.deallocate(r2, a);
  rm.deallocate(b);
  return 0;
}

