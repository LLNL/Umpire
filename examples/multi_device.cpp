//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_THREADS = 4096;
constexpr int MULTIPLE = 1024;

__global__ void touch_data(double* data, int len)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < len) {
    data[id] = id * MULTIPLE;
  }
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  std::cout << "Available allocators: ";
  for (auto s : rm.getResourceNames()) {
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  auto pool0 = rm.makeAllocator<umpire::strategy::QuickPool>("pool0", rm.getAllocator("DEVICE::0"));

  auto pool1 = rm.makeAllocator<umpire::strategy::QuickPool>("pool1", rm.getAllocator("DEVICE::1"));

  double* a = static_cast<double*>(pool0.allocate(NUM_THREADS * sizeof(double)));
  double* b = static_cast<double*>(pool1.allocate(NUM_THREADS * sizeof(double)));

  int NUM_BLOCKS = NUM_THREADS / BLOCK_SIZE;

#if defined(UMPIRE_ENABLE_CUDA)
  cudaSetDevice(0);
  touch_data<<<NUM_BLOCKS, BLOCK_SIZE>>>(a, NUM_THREADS);
  cudaDeviceSynchronize();
#endif
#if defined(UMPIRE_ENABLE_HIP)
  hipSetDevice(0);
  hipLaunchKernelGGL(touch_data, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, 0, a, NUM_THREADS);
  hipDeviceSynchronize();
#endif

  rm.copy(b, a);
  b = static_cast<double*>(rm.move(b, rm.getAllocator("HOST")));

  UMPIRE_ASSERT(b[BLOCK_SIZE] == (BLOCK_SIZE * MULTIPLE) && "Error: incorrect value!");
  std::cout << "Kernel succeeded! Expected result returned - " << b[BLOCK_SIZE] << std::endl;

  rm.deallocate(a);
  rm.deallocate(b);
}
