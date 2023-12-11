//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <stdio.h>

#include "camp/camp.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/StreamAwareQuickPool.hpp"

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_THREADS = 4096;

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

  //sleep
  sleep(100);

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
  auto pool = rm.makeAllocator<umpire::strategy::StreamAwareQuickPool>("sap-pool", rm.getAllocator("DEVICE"));

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  double* a = static_cast<double*>(pool.allocate(s1, NUM_THREADS * sizeof(double)));
  double* b = static_cast<double*>(pool.allocate(s2, NUM_THREADS * sizeof(double)));

  int NUM_BLOCKS = NUM_THREADS / BLOCK_SIZE;

  touch_data<<<NUM_BLOCKS, BLOCK_SIZE, 0, s1>>>(a, NUM_THREADS);
  touch_data_again<<<NUM_BLOCKS, BLOCK_SIZE, 0, s2>>>(a, NUM_THREADS);

  rm.copy(b, a);
  b = static_cast<double*>(rm.move(b, rm.getAllocator("HOST")));

  for (int i = 0; i < NUM_THREADS; i++) {
    UMPIRE_ASSERT(b[i] != (-1) && "Error: incorrect value!");
  }
  std::cout << "Kernel succeeded! Expected result returned - " << std::endl;

  rm.deallocate(a);
  rm.deallocate(b);
}
