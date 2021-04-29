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

constexpr int ITER {100}; //number of iterations, used for averaging time
constexpr int THREADS_PER_BLOCK {1024};

__global__ void first_in_block(umpire::DeviceAllocator alloc, double** data_ptr)
{
  for (int i = 0; i < ITER; i++) {
    if (threadIdx.x == 0) {
      double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
      *data_ptr = data;
      *data = 1024;
    }
  }
}

__global__ void only_first(umpire::DeviceAllocator alloc, double** data_ptr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < ITER; i++) {
    if (idx == 0) {
      double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
      *data_ptr = data;
      *data = 512;
    }
  }
}

__global__ void each_thread(umpire::DeviceAllocator alloc, double** data_ptr, unsigned int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < ITER; i++) {
    if (idx < N) {
      double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
      *data_ptr = data;
      *data = 256;
    }
  }
}

__global__ void warm_up(umpire::DeviceAllocator alloc, double** data_ptr, unsigned int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
    *data_ptr = data;
    *data = 42;
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

void event_timing_reporting(cudaEvent_t start, cudaEvent_t stop, double** ptr, unsigned long int total, std::string name)
{
  float milliseconds {0};
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  CudaTest("Checking for error just after kernel...");

  std::cout << name << std::endl;
  std::cout << "Time: " << (milliseconds/total*1000.0) << "us" << std::endl;
  std::cout << "Retrieved value: " << (*ptr)[0] << std::endl << std::endl;
}

int main(int, char**) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  double** ptr_to_data = static_cast<double**>(allocator.allocate(sizeof(double*)));

  unsigned int total_allocations {0};
  unsigned int N {0};

  cudaDeviceProp devProp;
  cudaSetDevice(0);
  cudaGetDeviceProperties(&devProp, 0);
  //TODO: fix me!
  if (devProp.concurrentKernels == 1) {
    N = 128 * THREADS_PER_BLOCK;
  } else {
    N = 32 * THREADS_PER_BLOCK;
  }
  std::cout<<"Running on device: "<<devProp.name<<std::endl;
  std::cout<<"Number of threads: "<<N<<std::endl;

  assert((N % THREADS_PER_BLOCK) != 0);

  //create cuda streams and events
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //Run warm-up kernel
  /////////////////////////////////////////////////
  total_allocations = N;
  auto dev_alloc_warmup = umpire::DeviceAllocator(allocator, (total_allocations) * sizeof(double));
  cudaEventRecord(start);
  warm_up<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(dev_alloc_warmup, ptr_to_data, N);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, total_allocations, "Kernel: Warm-up"); 
  /////////////////////////////////////////////////

  //Run kernel to allocate per first thread per block
  /////////////////////////////////////////////////
  total_allocations = (N/THREADS_PER_BLOCK*ITER);
  auto dev_alloc_firstInBlk = umpire::DeviceAllocator(allocator, (total_allocations) * sizeof(double));
  cudaEventRecord(start);
  first_in_block<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(dev_alloc_firstInBlk, ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, total_allocations, "Kernel: First thread per block"); 
  /////////////////////////////////////////////////

  //Run kernel to allocate with only thread 0
  /////////////////////////////////////////////////
  total_allocations = ITER;
  auto dev_alloc_onlyFirst = umpire::DeviceAllocator(allocator, (total_allocations) * sizeof(double));
  cudaEventRecord(start);
  only_first<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(dev_alloc_onlyFirst, ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, total_allocations, "Kernel: Only thread idx 0"); 
  /////////////////////////////////////////////////

  //Run kernel to allocate per each thread
  /////////////////////////////////////////////////
  total_allocations = N * ITER;
  auto dev_alloc_eachThread = umpire::DeviceAllocator(allocator, (total_allocations) * sizeof(double));
  cudaEventRecord(start);
  each_thread<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(dev_alloc_eachThread, ptr_to_data, N);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, total_allocations, "Kernel: Each thread"); 
  /////////////////////////////////////////////////

  cudaStreamDestroy(stream);
  return 0;
}
