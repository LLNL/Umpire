//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstdlib>
#include <time.h>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_THREADS = 8;
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

  auto ipool = rm.makeAllocator<umpire::strategy::QuickPool>("pool-input", rm.getAllocator("HOST"));
  auto rpool = rm.makeAllocator<umpire::strategy::QuickPool>("pool-result", rm.getAllocator("HOST"));

  int input_size = NUM_THREADS * sizeof(double);
  int sample_size = 4;
  int result_size = input_size-sample_size+1;

  double* input = static_cast<double*>(ipool.allocate(input_size));
  double* result = static_cast<double*>(rpool.allocate(result_size));

  //int NUM_BLOCKS = NUM_THREADS / BLOCK_SIZE;
  srand(time(0));
  for (int i = 0; i < input_size; i++) {
    input[i] = rand() % input_size;
  }

  std::cout<<"input:"<<std::endl;
  for (int i = 0; i < input_size; i++) {
    std::cout << input[i] << " ";
  }
  std::cout<<std::endl;

  for (int idx = 0; idx < result_size; idx++) {
    double sum = 0;
    for (int j = 0; j < sample_size; j++) {
      sum += input[idx+j];
    }

    sum /= sample_size;
    result[idx] = sum;
  }

  std::cout<<std::endl;
  std::cout<<std::endl;
  std::cout<<"RESULT:"<<std::endl;
  for (int i = 0; i < result_size; i++) {
    std::cout << result[i] << " ";
  }
  std::cout<<std::endl;

#if defined(UMPIRE_ENABLE_CUDA)
  //cudaSetDevice(0);
  //touch_data<<<NUM_BLOCKS, BLOCK_SIZE>>>(a, NUM_THREADS);
  //cudaDeviceSynchronize();
#endif
#if defined(UMPIRE_ENABLE_HIP)
  //hipSetDevice(0);
  //hipLaunchKernelGGL(touch_data, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, 0, a, NUM_THREADS);
  //hipDeviceSynchronize();
#endif

  //rm.copy(b, a);
  //b = static_cast<double*>(rm.move(b, rm.getAllocator("HOST")));

  //UMPIRE_ASSERT(b[BLOCK_SIZE] == (BLOCK_SIZE * MULTIPLE) && "Error: incorrect value!");
  //std::cout << "Kernel succeeded! Expected result returned - " << b[BLOCK_SIZE] << std::endl;

  //rm.deallocate(a);
  //rm.deallocate(b);
  ipool.deallocate(input);
  rpool.deallocate(result);
}
