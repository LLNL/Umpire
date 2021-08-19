//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/device_allocator_helper.hpp"
#include "umpire/ResourceManager.hpp"

/*
 * Very simple kernel that uses only the first thread to "get" the
 * existing DeviceAllocator and to allocate an array of ints.
 * Making sure that the data_ptr is pointing to the device allocated array,
 * it sets one element to a value that will be checked later.
 */
__global__ void my_kernel(double** data_ptr)
{
  if (threadIdx.x == 0) {
    umpire::DeviceAllocator alloc = umpire::util::getDeviceAllocator(0);
    double* data = static_cast<double*>(alloc.allocate(10 * sizeof(double)));
    *data_ptr = data;
    data[7] = 1024;
  }
}

int main(int argc, char const* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();

  //Checking to make sure a DeviceAllocator doesn't yet exist
  if(umpire::util::existsDeviceAllocator) {
    std::cout << "Before I create a DeviceAllocator, it doesn't exist!" << std::endl;
  }

  //Create all of my allocators
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = rm.makeDeviceAllocator(allocator, 1024);
  auto device_allocator2 = rm.makeDeviceAllocator(allocator, 2048);

  //Checking that now a DeviceAllocator exists
  if(umpire::util::existsDeviceAllocator)
    std::cout << "I found a DeviceAllocator!" << std::endl;

  double** ptr_to_data =
      static_cast<double**>(allocator.allocate(sizeof(double*)));

  //Make sure that device and host side DeviceAllocator pointers are synched
  UMPIRE_SET_UP_DEVICE_ALLOCATOR_ARRAY();

  my_kernel<<<1, 16>>>(ptr_to_data);
  cudaDeviceSynchronize();

  //Printing out the kernel result, plus the IDs for my DeviceAllocators
  std::cout << "Found value: " << (*ptr_to_data)[7] << std::endl;
  std::cout << "DA1 with ID: " << device_allocator.getID() << std::endl;
  std::cout << "DA2 with ID: " << device_allocator2.getID() << std::endl;

  return 0;
}
