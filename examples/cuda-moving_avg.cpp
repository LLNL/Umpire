//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

constexpr int NUM_THREADS = 256;

__global__ void moving_avg(double* input, double* result, int result_size, int sample_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < result_size) {
    double sum = 0;
    for (int j = 0; j < sample_size; j++) {
      sum += input[idx+j];
    }

    sum /= sample_size;
    result[idx] = sum;
  }
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  struct timeval start, end;
  double* input{nullptr};
  double* result{nullptr};

  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool", rm.getAllocator("UM"));

  int input_size{1<<28};
  int sample_size{256};
  int result_size = input_size-sample_size+1;

  input = static_cast<double*>(pool.allocate(input_size*sizeof(double)));
  result = static_cast<double*>(pool.allocate(result_size*sizeof(double)));

  srand(time(0));
  for (int i = 0; i < input_size; i++) {
    input[i] = rand() % input_size;
  }

  gettimeofday(&start, NULL);
  moving_avg<<<((result_size + NUM_THREADS-1) / NUM_THREADS), NUM_THREADS>>>(input, result, result_size, sample_size);
  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);

  printf("compute time: %.8f s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);

  pool.deallocate(input);
  pool.deallocate(result);
}
