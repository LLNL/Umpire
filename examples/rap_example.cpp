#include <stdio.h>

#include <iostream>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"

using namespace camp::resources;

#if defined(UMPIRE_ENABLE_CUDA)
using resource_type = Cuda;
#elif defined(UMPIRE_ENABLE_HIP)
using resource_type = Hip;
#else
using resource_type = Host;
#endif

constexpr int NUM_THREADS = 64;

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

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
constexpr int BLOCK_SIZE = 16;
using clock_value_t = long long;

#if defined(UMPIRE_ENABLE_CUDA)
__device__ clock_value_t my_clock()
{
  return clock64();
}
#elif defined(UMPIRE_ENABLE_HIP)
__device__ clock_value_t my_clock()
{
  return hipGetClock();
}
#endif

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
  sleep(10000000);
}
#endif

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("UM"));
  const int NUM_BLOCKS = NUM_THREADS / BLOCK_SIZE;
#else
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("HOST"));
#endif

  resource_type d1, d2, d3;
  Resource r1{d1}, r2{d2}, r3{d3};

  // allocate memory in the pool with r1
  double* a = static_cast<double*>(pool.allocate(r1, NUM_THREADS * sizeof(double)));
  double* ptr1 = a;

  // Make sure resource was correctly tracked
  UMPIRE_ASSERT(getResource(pool, a) == r1);

  // launch kernels on r1's stream
#if defined(UMPIRE_ENABLE_CUDA)
  do_sleep<<<NUM_BLOCKS, BLOCK_SIZE, 0, d1.get_stream()>>>();
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(do_sleep, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, d1.get_stream());
#else
  host_sleep(a);
#endif

  // deallocate memory with r1 and reallocate using a different stream r2
  pool.deallocate(r1, a);

  a = static_cast<double*>(pool.allocate(r2, NUM_THREADS * sizeof(double)));
  double* ptr2 = a;

  UMPIRE_ASSERT(getResource(pool, a) == r2);

  // Use Camp resource to synchronize devices
  r1.get_event().wait();

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
  UMPIRE_ASSERT(ptr1 != ptr2);
#else
  UMPIRE_ASSERT(ptr1 == ptr2);
#endif
  std::cout << "Expected result returned!" << std::endl;

  // deallocate and clean up
  pool.deallocate(r2, a);
  return 0;
}
