//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"

__global__ void touch_data(double* data, int len)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < len) {
    data[id] = id * 1024;
  }
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();

  std::cout << "Available allocators: ";
  for (auto s : rm.getAllocatorNames()) {
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  auto pool0 = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "pool0", rm.getAllocator("DEVICE_0"));

  auto pool1 = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "pool1", rm.getAllocator("DEVICE_1"));

  double* a = static_cast<double*>(pool0.allocate(4096 * sizeof(double)));
  double* b = static_cast<double*>(pool1.allocate(4096 * sizeof(double)));

  int BLOCK_SIZE = 256;
  int NUM_BLOCKS = 4096 / 256;

  cudaSetDevice(0);
  touch_data<<<NUM_BLOCKS, BLOCK_SIZE>>>(a, 4096);
  cudaDeviceSynchronize();

  rm.copy(b, a);

  b = static_cast<double*>(rm.move(b, rm.getAllocator("HOST")));
  std::cout << "a[256]= " << b[256] << std::endl;

  rm.deallocate(a);
  rm.deallocate(b);
}
