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

void do_stuff(double* a)
{
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
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool", rm.getAllocator("DEVICE"));

  double* a = static_cast<double*>(pool.allocate(NUM_THREADS * sizeof(double)));
  double* b = static_cast<double*>(pool.allocate(NUM_THREADS * sizeof(double)));

  do_stuff(a);

  rm.copy(b, a);
  b = static_cast<double*>(rm.move(b, rm.getAllocator("HOST")));
  UMPIRE_ASSERT(b[BLOCK_SIZE] == (BLOCK_SIZE * MULTIPLE) && "Error: incorrect value!");
  std::cout << "Kernel succeeded! Expected result returned - " << b[BLOCK_SIZE] << std::endl;

  pool.deallocate(a);
  a = static_cast<double*>(pool.allocate(NUM_THREADS * sizeof(double)));
  do_stuff(a);

  pool.deallocate(a);
  a = static_cast<double*>(pool.allocate(NUM_THREADS * NUM_THREADS * sizeof(double)));

  rm.deallocate(a);
  rm.deallocate(b);
}
