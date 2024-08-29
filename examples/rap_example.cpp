#include <stdio.h>

#include <iostream>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"
#include "umpire/strategy/QuickPool.hpp"

constexpr int NUM_THREADS = 64;

using clock_value_t = long long;
using namespace camp::resources;

void host_touch_data(double* ptr)
{
  for(int i = 0; i < NUM_THREADS; i++) {
    ptr[i] = i;
  }
}

void host_touch_data_again(double* ptr)
{
  for(int i = 0; i < NUM_THREADS; i++) {
    ptr[i] = 54321;
  }
}

void host_check_data(double* ptr)
{
  for(int i = 0; i < NUM_THREADS; i++) {
    if(ptr[i] != i) {
      ptr[i] = -1;
    }
  }
}

void host_sleep(double* ptr)
{
  double i = 0.0;
  while (i < 1000000) {
    double y = i;
    y++;
    i = y;
  }
  *ptr = i;
  ptr++;
}

#if defined(UMPIRE_ENABLE_DEVICE)
constexpr int BLOCK_SIZE = 16;

__device__ void sleep(clock_value_t sleep_cycles)
{
  clock_value_t start = clock64();
  clock_value_t cycles_elapsed;
  do {
    cycles_elapsed = clock64() - start;
  } while (cycles_elapsed < sleep_cycles);
}

__global__ void touch_data(double* data, int len)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < len) {
    data[id] = 1.0 * id;
  }
}

__global__ void do_sleep()
{
  // sleep - works still at 1000, so keeping it at 100k
  sleep(10000000);
}

__global__ void check_data(double* data, int len)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Then error check that data[id] still == id
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
#endif

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
#if defined(UMPIRE_ENABLE_DEVICE)
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("UM"));
  const int NUM_BLOCKS = NUM_THREADS / BLOCK_SIZE;
#else
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("HOST"));
#endif

  // Create camp resources
#if defined(UMPIRE_ENABLE_CUDA)
  Cuda d1, d2, d3;
#elif defined(UMPIRE_ENABLE_HIP)
  Hip d1, d2, d3;
#else
  Host d1, d2, d3;
#endif
  Resource r1{d1}, r2{d2}, r3{d3};

  // allocate memory in the pool with r1
  double* a = static_cast<double*>(pool.allocate(r1, NUM_THREADS * sizeof(double)));
  double* ptr1 = a;

  // Make sure resource was correctly tracked
  UMPIRE_ASSERT(getResource(pool, a) == r1);

  // launch kernels on r1's stream
#if defined(UMPIRE_ENABLE_CUDA)
  touch_data<<<NUM_BLOCKS, BLOCK_SIZE, 0, d1.get_stream()>>>(a, NUM_THREADS);
  do_sleep<<<NUM_BLOCKS, BLOCK_SIZE, 0, d1.get_stream()>>>();
  check_data<<<NUM_BLOCKS, BLOCK_SIZE, 0, d1.get_stream()>>>(a, NUM_THREADS);
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(touch_data, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, d1.get_stream(), a, NUM_THREADS);
  hipLaunchKernelGGL(do_sleep, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, d1.get_stream());
  hipLaunchKernelGGL(check_data, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, d1.get_stream(), a, NUM_THREADS);
#else
  host_touch_data(a);
  host_sleep(a);
  host_check_data(a);
#endif

  // deallocate memory with r1 and reallocate using a different stream r2
  pool.deallocate(r1, a);

  a = static_cast<double*>(pool.allocate(r2, NUM_THREADS * sizeof(double)));
  double* ptr2 = a;

  UMPIRE_ASSERT(getResource(pool, a) == r2);

  // launch kernel with r2's stream using newly reallocated 'a'
#if defined(UMPIRE_ENABLE_CUDA)
  touch_data_again<<<NUM_BLOCKS, BLOCK_SIZE, 0, d2.get_stream()>>>(a, NUM_THREADS);
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(touch_data_again, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, d2.get_stream(), a, NUM_THREADS);
#else
  host_touch_data_again(a);
#endif

  // For validation/error checking below, use camp resource to synchronize host and device
#if defined(UMPIRE_ENABLE_CUDA)
  cudaDeviceSynchronize();
#elif defined(UMPIRE_ENABLE_HIP)
  hipDeviceSynchronize();
#endif

  // Error check and validation
  for (int i = 0; i < NUM_THREADS; i++) {
    UMPIRE_ASSERT(a[i] != (-1) && "Error: incorrect value!");
  }
#if defined(UMPIRE_ENABLE_DEVICE)
  UMPIRE_ASSERT(ptr1 != ptr2);
#else
  UMPIRE_ASSERT(ptr1 == ptr2);
#endif
  std::cout << "Kernel succeeded! Expected result returned" << std::endl;

  // deallocate and clean up
  pool.deallocate(r2, a);
  return 0;
}
