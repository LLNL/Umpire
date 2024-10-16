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
std::cout<<"Made it past allocation1"<<std::endl;

    hipLaunchKernelGGL(init, dim3(NUM_BLOCKS), dim3(4), 0, d1.get_stream(), b, c);
std::cout<<"Made it past init kernel"<<std::endl;
    hipDeviceSynchronize();
    hipLaunchKernelGGL(body, dim3(NUM_BLOCKS), dim3(4), 0, d1.get_stream(), a, b, c);
std::cout<<"Made it past kernel launch1"<<std::endl;

    pool.deallocate(a);
    pool.deallocate(b);
    pool.deallocate(c);
std::cout<<"Made it past deallocation1"<<std::endl;

    a = static_cast<double*>(pool.allocate(r2, NUM * sizeof(double)));
    b = static_cast<double*>(pool.allocate(r2, NUM * sizeof(double)));
    c = static_cast<double*>(pool.allocate(r2, NUM * sizeof(double)));
std::cout<<"Made it past allocation2"<<std::endl;

    hipLaunchKernelGGL(init, dim3(NUM_BLOCKS), dim3(4), 0, d2.get_stream(), b, c);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(body, dim3(NUM_BLOCKS), dim3(4), 0, d2.get_stream(), a, b, c);
std::cout<<"Made it past kernel launch2"<<std::endl;

    hipDeviceSynchronize();
    check(a, b, c);
std::cout<<"Made it past the check"<<std::endl;

    pool.deallocate(a);
std::cout<<"Made it past deallocation of a"<<std::endl;
    pool.deallocate(b);
std::cout<<"Made it past deallocation of b"<<std::endl;
    pool.deallocate(c);
std::cout<<"Made it past deallocation2"<<std::endl;

  }

  //pool.release();
  return 0;
}
  
