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

__global__ void each_thread(double** data_ptr, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    auto alloc = umpire::get_device_allocator("dev_alloc");
    double* data = static_cast<double*>(alloc.allocate(sizeof(double)));
    *data_ptr = data;
    *data = 256;
  }
}

#if defined(UMPIRE_ENABLE_CUDA)
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
#elif defined(UMPIRE_ENABLE_HIP)
static void HipTest(const char *msg)
{
  hipError_t e = hipGetLastError();
  hipDeviceSynchronize();
  if (hipSuccess != e) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", hipGetErrorString(e));
    exit(-1);
  }
}
#endif

#if defined(UMPIRE_ENABLE_CUDA)
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
#elif defined(UMPIRE_ENABLE_HIP)
void event_timing_reporting(hipEvent_t start, hipEvent_t stop, double** ptr, unsigned int total, std::string name)
{
  float milliseconds {0};
  hipEventSynchronize(stop);
  hipEventElapsedTime(&milliseconds, start, stop);
  HipTest("Checking for error just after kernel...\n");

  std::cout << name << std::endl;
  std::cout << "Total time: " << (milliseconds*1000.0) << "us" << std::endl;
  std::cout << "Time per allocation: " << (milliseconds/total*1000.0) << "us" << std::endl;
  std::cout << "Retrieved value: " << *ptr[0] << std::endl << std::endl;
}
#endif

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  double** ptr_to_data = static_cast<double**>(allocator.allocate(sizeof(double*)));

  int N {0};
  N = NUM_SM * BLOCKS_PER_SM * THREADS_PER_BLOCK; 

  // Set up the device and get properties
#if defined(UMPIRE_ENABLE_CUDA)
  cudaDeviceProp devProp;
  cudaSetDevice(0);
  cudaGetDeviceProperties(&devProp, 0);
#elif defined(UMPIRE_ENABLE_HIP)
  hipDeviceProp_t devProp;
  hipSetDevice(0);
  hipGetDeviceProperties(&devProp, 0);
#endif
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

  // Create streams and events
#if defined(UMPIRE_ENABLE_CUDA)
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#elif defined(UMPIRE_ENABLE_HIP)
  hipStream_t stream;
  hipStreamCreate(&stream);
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
#endif

  // Run warm-up kernel
  //////////////////////////////////////////////////
#if defined(UMPIRE_ENABLE_CUDA)
  cudaEventRecord(start);
  only_first<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, 1, "Kernel: Warm-up"); 
#elif defined(UMPIRE_ENABLE_HIP)
  hipEventRecord(start);
  hipLaunchKernelGGL(only_first, dim3(N/THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, stream, ptr_to_data);
  hipEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, 1, "Kernel: Warm-up"); 
#endif
  //////////////////////////////////////////////////

  dev_alloc.reset();

  // Run kernel to allocate per first thread per block
  //////////////////////////////////////////////////
#if defined(UMPIRE_ENABLE_CUDA)
  cudaEventRecord(start);
  first_in_block<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, (N/THREADS_PER_BLOCK), "Kernel: First thread per block");
#elif defined(UMPIRE_ENABLE_HIP)
  hipEventRecord(start);
  hipLaunchKernelGGL(first_in_block, dim3(N/THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, stream, ptr_to_data);
  hipEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, (N/THREADS_PER_BLOCK), "Kernel: First thread per block");
#endif
  //////////////////////////////////////////////////

  dev_alloc.reset();

  // Run kernel to allocate with only thread 0
  //////////////////////////////////////////////////
#if defined(UMPIRE_ENABLE_CUDA)
  cudaEventRecord(start);
  only_first<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(ptr_to_data);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, 1, "Kernel: Only thread idx 0");
#elif defined(UMPIRE_ENABLE_HIP)
  hipEventRecord(start);
  hipLaunchKernelGGL(only_first, dim3(N/THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, stream, ptr_to_data);
  hipEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, 1, "Kernel: Only thread idx 0");
#endif
  //////////////////////////////////////////////////

  dev_alloc.reset();

  // Run kernel to allocate per each thread
  //////////////////////////////////////////////////
#if defined(UMPIRE_ENABLE_CUDA)
  cudaEventRecord(start);
  each_thread<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(ptr_to_data, N);
  cudaEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, N, "Kernel: Each thread"); 
#elif defined(UMPIRE_ENABLE_HIP)
  hipEventRecord(start);
  hipLaunchKernelGGL(each_thread, dim3(N/THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, stream, ptr_to_data, N);
  hipEventRecord(stop);
  event_timing_reporting(start, stop, ptr_to_data, N, "Kernel: Each thread"); 
#endif
  //////////////////////////////////////////////////

#if defined(UMPIRE_ENABLE_CUDA)
  cudaFree(ptr_to_data);
  cudaStreamDestroy(stream);
#elif defined(UMPIRE_ENABLE_HIP)
  hipFree(ptr_to_data);
  hipStreamDestroy(stream);
#endif
  return 0;
}
