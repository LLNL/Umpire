//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <stdio.h>

#include "umpire/util/device_allocator_helper.hpp"
#include "umpire/ResourceManager.hpp"

const int THREADS_PER_BLOCK = 128;
const int BLOCKS = 4;
__device__ int* data{nullptr};

__global__ void my_kernel(int* result)
{
  unsigned int i{threadIdx.x+blockIdx.x*blockDim.x};

  if (i == 0) {
    umpire::DeviceAllocator alloc = umpire::util::getDeviceAllocator(0);
    data = static_cast<int*>(alloc.allocate(BLOCKS * sizeof(int)));
  }

  data[blockIdx.x] = blockIdx.x;
  result[i] = BLOCKS * data[blockIdx.x];
}

int main(int argc, char const* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = rm.makeDeviceAllocator(allocator, BLOCKS*sizeof(int));

  int* result = static_cast<int*>(allocator.allocate(BLOCKS*THREADS_PER_BLOCK * sizeof(int)));
  memset(result, 0, BLOCKS*THREADS_PER_BLOCK);

  UMPIRE_SET_UP_DEVICE_ALLOCATOR_ARRAY();
  my_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(result);
  cudaDeviceSynchronize();

  std::cout << "Result: " <<std::endl;
  for(int i = 0; i < THREADS_PER_BLOCK; i++)
    for(int j = 0; j < BLOCKS; j++)
      std::cout << result[i*BLOCKS+j] << " ";
  std::cout << std::endl;
  std::cout << std::endl;
  for(int i = 0; i < BLOCKS; i++)
    std::cout << result[i*THREADS_PER_BLOCK] << " ";
  std::cout << std::endl;
  std::cout << std::endl;
  for(int i = 0; i < THREADS_PER_BLOCK; i++) {
    for(int j = 0; j < BLOCKS; j++) {
      if (result[i*BLOCKS+j] != j*BLOCKS)
        std::cout << "ERROR detected!" << std::endl;
    }
  }

  return 0;
}
