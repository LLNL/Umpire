////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
///////////////////////////////////////////////////////////////////////////
#include <chrono>
#include <assert.h>

#include "umpire/ResourceManager.hpp"
#include "umpire/DeviceAllocator.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/device_allocator_helper.hpp"

// Set device details given LC gpu compute capability
constexpr int THREADS_PER_BLOCK {1024};
constexpr int BLOCKS_PER_SM {32}; 
constexpr int NUM_SM {80}; 

__global__ void first_in_block(double** data_ptr)
{
  if (threadIdx.x == 0) {
    auto alloc = umpire::get_device_allocator("dev_alloc");
    double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
    *data_ptr = data;
    *data = 1024;
  }
}

__global__ void only_first(double** data_ptr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    auto alloc = umpire::get_device_allocator("dev_alloc");
    double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
    *data_ptr = data;
    *data = 512;
  }
}

__global__ void each_thread(double** data_ptr, unsigned int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    auto alloc = umpire::get_device_allocator("dev_alloc");
    double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
    *data_ptr = data;
    *data = 256;
  }
}

static void CudaTest(const char *msg)
{
  cudaError_t e = cudaGetLastError();
  cudaDeviceSynchronize();
  if (cudaSuccess != e) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}

void event_timing_reporting(cudaEvent_t start, cudaEvent_t stop, double** ptr, unsigned int total, std::string name)
{
  float milliseconds {0};
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  CudaTest("Checking for error just after kernel...\n");

  std::cout << name << std::endl;
  std::cout << "Total time: " << (milliseconds*1000.0) << "us" << std::endl;
  std::cout << "Time per allocation: " << (milliseconds/total*1000.0) << "us" << std::endl;
  std::cout << "Retrieved value: " << *ptr[0] << std::endl << std::endl;
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  double** ptr_to_data = static_cast<double**>(allocator.allocate(sizeof(double*)));

  unsigned int N {0};
  N = NUM_SM * BLOCKS_PER_SM * THREADS_PER_BLOCK; 

  // Set up the device and get properties
  cudaDeviceProp devProp;
  cudaSetDevice(0);
  cudaGetDeviceProperties(&devProp, 0);
  std::cout << "Running on device: " << devProp.name << std::endl;
  std::cout << "Number of threads: " << N << std::endl << std::endl;

  // Create device allocator and set up memory
  if (devProp.concurrentKernels != 1) {
    std::cout << std::endl << "**Current device does not support concurrent kernels. " << 
                 "Timing won't be as accurate. Continuing anyways... **" << std::endl;
  }

  // Create device allocator
  auto dev_alloc = umpire::make_device_allocator(allocator, N * sizeof(double), "dev_alloc");
  UMPIRE_SET_UP_DEVICE_ALLOCATORS(); // Still required in case this is called on different translation unit.

  // Create cuda streams and events
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Run warm-up kernel
  //////////////////////////////////////////////////
  cudaEventRecord(start);
  only_first<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, 1, "Kernel: Warm-up"); 
  //////////////////////////////////////////////////

  dev_alloc.reset();

  // Run kernel to allocate per first thread per block
  //////////////////////////////////////////////////
  cudaEventRecord(start);
  first_in_block<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, (N/THREADS_PER_BLOCK), "Kernel: First thread per block");
  //////////////////////////////////////////////////

  dev_alloc.reset();

  // Run kernel to allocate with only thread 0
  //////////////////////////////////////////////////
  cudaEventRecord(start);
  only_first<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, 1, "Kernel: Only thread idx 0");
  //////////////////////////////////////////////////

  dev_alloc.reset();

  // Run kernel to allocate per each thread
  //////////////////////////////////////////////////
  cudaEventRecord(start);
  each_thread<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(ptr_to_data, N);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, N, "Kernel: Each thread"); 
  //////////////////////////////////////////////////

  cudaFree(ptr_to_data);
  cudaStreamDestroy(stream);
  return 0;
}
