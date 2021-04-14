//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <chrono>
#include <assert.h>
#include "umpire/ResourceManager.hpp"
#include "umpire/DeviceAllocator.hpp"
#include "umpire/Allocator.hpp"

//constexpr int ALLOCATIONS {1<<10}; //testing over const number of allocations (1024)
constexpr int N {1<<10};
constexpr int THREADS_PER_BLOCK {256};

__global__ void one_per_block(umpire::DeviceAllocator alloc, double** data_ptr)
{
  if (threadIdx.x == 0) {
    double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
    *data_ptr = data;
    *data = 1024;
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
    *data = 256;
  }
}

static void CudaTest(const char *msg)
{
  cudaError_t e = cudaGetLastError();
  cudaThreadSynchronize();
  if (cudaSuccess != e) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}

void event_timing_reporting(cudaEvent_t start, cudaEvent_t stop, double** ptr)
{
  float milliseconds {0};
  CudaTest("Checking for error just after kernel: ");

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Time: " << milliseconds << "ms" << std::endl;
  std::cout << "Retrieved value: " << (*ptr)[0] << std::endl;
}

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = umpire::DeviceAllocator(allocator, N * sizeof(double));
  double** ptr_to_data =
      static_cast<double**>(allocator.allocate(sizeof(double*)));

  assert((N % THREADS_PER_BLOCK) != 0);

  //create cuda streams and events
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //Run kernel to allocate per first thread per block
  cudaEventRecord(start);
  one_per_block<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(device_allocator, ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data); 

  //Run kernel to allocate with only thread 0
  cudaEventRecord(start);
  only_the_first<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(device_allocator, ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data); 

  //Run kernel to allocate per each thread
  cudaEventRecord(start);
  each_one<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(device_allocator, ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data); 

  cudaStreamDestroy(stream);
  return 0;
}
