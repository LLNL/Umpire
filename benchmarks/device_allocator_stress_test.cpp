//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <chrono>
#include "umpire/ResourceManager.hpp"
#include "umpire/DeviceAllocator.hpp"
#include "umpire/Allocator.hpp"

//constexpr int ALLOCATIONS {1<<10}; //testing over const number of allocations (1024)
constexpr int N {1<<10};
constexpr int NUM_THREADS {256};

__global__ void one_per_block(umpire::DeviceAllocator alloc, double** data_ptr)
{
  if (threadIdx.x == 0) {
    double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
    *data_ptr = data;

    data[0] = 1024;
  }
}

__global__ void only_the_first(umpire::DeviceAllocator alloc, double** data_ptr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
    *data_ptr = data;

    data[0] = 512;
  }
}

__global__ void each_one(umpire::DeviceAllocator alloc, double** data_ptr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1019) {
    double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
    *data_ptr = data;

    data[0] = 256;
  }
}

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = umpire::DeviceAllocator(allocator, N * sizeof(double));

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  cudaError_t err;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double** ptr_to_data =
      static_cast<double**>(allocator.allocate(sizeof(double*)));

  cudaEventRecord(start);
  one_per_block<<<(N+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS, 0, stream1>>>(device_allocator, ptr_to_data);
  cudaEventRecord(stop);

  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
  }

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  cudaStreamDestroy(stream1);
  std::cout << "Retrieved value: " << (*ptr_to_data)[0] << std::endl;
  std::cout << "Time: " << milliseconds << "ms" << std::endl;

  cudaEventRecord(start);
  only_the_first<<<(N+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS, 0, stream2>>>(device_allocator, ptr_to_data);
  cudaEventRecord(stop);

  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
  }

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  cudaStreamDestroy(stream2);
  std::cout << "Retrieved value: " << (*ptr_to_data)[0] << std::endl;
  std::cout << "Time: " << milliseconds << "ms" << std::endl;

  cudaEventRecord(start);
  each_one<<<(N+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS, 0, stream3>>>(device_allocator, ptr_to_data);
  cudaEventRecord(stop);

  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
  }

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  cudaStreamDestroy(stream3);
  std::cout << "Retrieved value: " << (*ptr_to_data)[0] << std::endl;
  std::cout << "Time: " << milliseconds << "ms" << std::endl;

  return 0;
}
