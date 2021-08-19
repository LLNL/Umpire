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

/*
 * Kernel that performs simple arithmetic and stores the
 * result in the array of ints passed to the kernel.
 *
 * Note: Benign data race that writes to the device allocated
 * array, data.
 */
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

/*
 * Function to check if array of ints matches up against expected
 * results returned from kernel. 
 *
 * If error detected, returns true. Otherwise, returns false.
 */
bool checkIfErrorsDetected(int* results)
{
  for(int i = 0; i < BLOCKS; i++) {
    for(int j = 0; j < THREADS_PER_BLOCK; j++) {
      if (results[i*THREADS_PER_BLOCK+j] != i*BLOCKS) {
        return true;
      }
    }
  }
  return false;
}

/*
 * Function that performs the body of the test, then calls the
 * checkIfErrorsDetected function to check results.
 *
 * Note: This function is called from main and does not have any
 * record of a DeviceAllocator object. This is by design to show that
 * a function which is called from main can then call a kernel to
 * perform some test with a DeviceAllocator object. The kernel will 
 * still have access to any DeviceAllocator object created in main,
 * even if it is not explicitly passed to this intermediate function.
 */
void performTest(umpire::Allocator allocator)
{
  //create an array of integers which will be used by the gpu to do some simple arithmetic
  int* results = static_cast<int*>(allocator.allocate(BLOCKS*THREADS_PER_BLOCK * sizeof(int)));
  memset(results, 0, BLOCKS*THREADS_PER_BLOCK);

  //Sync up the device and host side global pointers that keep track of DeviceAllocator objects
  UMPIRE_SET_UP_DEVICE_ALLOCATOR_ARRAY();

  //call kernel and sync
  my_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(results);
  cudaDeviceSynchronize();
  
  std::cout << "Checking results..." <<std::endl;
  if (checkIfErrorsDetected(results)) {
    std::cout << "There were errors detected!" << std::endl;
  }
  else {
    std::cout << "SUCCESS!" << std::endl;
  }
}

int main(int argc, char const* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();

  //create the allocators that will be used in this program
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = rm.makeDeviceAllocator(allocator, BLOCKS*sizeof(int));

  //call the function that performs the actual test
  performTest(allocator);
  return 0;
}
