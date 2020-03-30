
#include <caliper/cali.h>

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include <cstring>
#include <iostream>
#include <string>

constexpr int COUNT  = 100000;
constexpr int BLOCK_SIZE = 64;

__global__
void daxpy(double* x, double* y, double a)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < COUNT)
    y[i]] += x[i] * a;
}

int main(int argc, char* argv[])
{
    auto& rm = umpire::ResourceManager::getInstance();
    auto allocator = rm.getAllocator("UM");

    double* x = static_cast<double*>{allocator.allocate(COUNT*sizeof(double))};
    double* y = static_cast<double*>{allocator.allocate(COUNT*sizeof(double))};

    for (auto i = 0; i < COUNT; i++) {
        x[i] = 1.0;
        y[i] = 0.0;
    }

    double a{3.14};

    daxpy<<<(COUNT/BLOCK_SIZE) + 1, BLOCK_SIZE>>>(x, y, a);

    cudaDeviceSynchronize();
}