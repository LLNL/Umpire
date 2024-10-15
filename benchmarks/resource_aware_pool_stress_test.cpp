#include <stdio.h>
#include <iostream>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"

using namespace camp::resources;

constexpr int NUM = 32;
constexpr double alpha = 3.14;
const int NUM_BLOCKS = NUM / 4;

__global__ void init (double* b, double* c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < NUM) {
    b[i] = i;
    c[i] = i;
  }
}

__global__ void body (double* a, double* b, double* c)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < NUM) {
    a[i] = b[i] + alpha * c[i];
  }
}

void check(double* a, double* b, double* c)
{
  for (int i = 0; i < NUM; i++) {
    if(b[i] != i) {std::cerr << "Kernel error occurred with b: " << b[i] << " at " << i << std::endl;}
    if(c[i] != i) {std::cerr << "Kernel error occurred with c: " << c[i] << " at " << i << std::endl;}
    if(a[i] != (b[i] + alpha * c[i])) {std::cerr << "Kernel error occurred with a: " << a[i] << " instead of: " << (b[i] + alpha * c[i]) << " at " << i << std::endl;}
  }
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("UM"));

  Hip d1, d2;
  Resource r1{d1}, r2{d2};

  for(int i = 0; i < 5; i++) {
    double* a = static_cast<double*>(pool.allocate(r1, NUM * sizeof(double)));
    double* b = static_cast<double*>(pool.allocate(r1, NUM * sizeof(double)));
    double* c = static_cast<double*>(pool.allocate(r1, NUM * sizeof(double)));

    hipLaunchKernelGGL(init, dim3(NUM_BLOCKS), dim3(4), 0, d1.get_stream(), b, c);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(body, dim3(NUM_BLOCKS), dim3(4), 0, d1.get_stream(), a, b, c);

    pool.deallocate(a);
    pool.deallocate(b);
    pool.deallocate(c);

    a = static_cast<double*>(pool.allocate(r2, NUM * sizeof(double)));
    b = static_cast<double*>(pool.allocate(r2, NUM * sizeof(double)));
    c = static_cast<double*>(pool.allocate(r2, NUM * sizeof(double)));

    hipLaunchKernelGGL(init, dim3(NUM_BLOCKS), dim3(4), 0, d2.get_stream(), b, c);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(body, dim3(NUM_BLOCKS), dim3(4), 0, d2.get_stream(), a, b, c);

    hipDeviceSynchronize();
    check(a, b, c);

    pool.deallocate(a);
    pool.deallocate(b);
    pool.deallocate(c);

  }

  //pool.release();
  return 0;
}
  
