/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <sys/time.h>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/device_allocator_helper.hpp"

constexpr int NUM_THREADS = 256;
__device__ double** input;

__global__ void init_input(int input_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t state;
  curand_init(input_size, 0, idx, &state);

  if (idx == 0) {
    umpire::DeviceAllocator alloc = umpire::get_device_allocator("my_device_alloc");
    input = static_cast<double*>(alloc.allocate(input_size*sizeof(double)));
  }

  if (idx < input_size) {
    input[idx] = curand(&state) % 100;
  }
}

__global__ void moving_avg(double* result, int result_size, int sample_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //if (idx == 0) {
  // umpire::DeviceAllocator alloc = umpire::get_device_allocator("my_device_alloc");

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
  double* result{nullptr};

  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("pool", rm.getAllocator("UM"));

  int input_size{1<<16};
  int sample_size{256};
  //int input_size = 8; int sample_size = 2;
  int result_size = input_size-sample_size+1;

  auto device_allocator = umpire::make_device_allocator(rm.getAllocator("UM"), input_size*sizeof(double), "my_device_alloc");
  result = static_cast<double*>(pool.allocate(result_size*sizeof(double)));

  UMPIRE_SET_UP_DEVICE_ALLOCATORS();

  gettimeofday(&start, NULL);

  init_input<<<((input_size + NUM_THREADS-1) / NUM_THREADS), NUM_THREADS>>>(input_size);
  cudaDeviceSynchronize();
  moving_avg<<<((result_size + NUM_THREADS-1) / NUM_THREADS), NUM_THREADS>>>(result, result_size, sample_size);
  cudaDeviceSynchronize();

  gettimeofday(&end, NULL);
  printf("compute time: %.8f s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);

  pool.deallocate(result);
}
