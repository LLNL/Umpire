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

  // Create camp resources for RAP
  resource_type d1, d2;

  // allocate memory in the pool with d1
  double* a = static_cast<double*>(pool.allocate(NUM_THREADS * sizeof(double), d1));
  double* ptr1 = a;

  // launch kernels on d1's stream
#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
  do_sleep<<<NUM_BLOCKS, BLOCK_SIZE, 0, d1.get_stream()>>>();
#else
  host_sleep(a);
#endif

  // deallocate memory with d1 and reallocate using a different stream d2
  pool.deallocate(a, d1); // Deallocate using resource
  a = static_cast<double*>(pool.allocate(NUM_THREADS * sizeof(double), d2));
  double* ptr2 = a;

  // Use Camp resource to synchronize devices
  d2.get_event().wait();

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
  UMPIRE_ASSERT(ptr1 != ptr2);
#else
  UMPIRE_ASSERT(ptr1 == ptr2);
#endif
  std::cout << "Expected result returned! Ptr1 = " << ptr1 << "; Ptr2 = " << ptr2 << ";" << std::endl;

  pool.deallocate(a); // Deallocation with no resource included
  return 0;
}
