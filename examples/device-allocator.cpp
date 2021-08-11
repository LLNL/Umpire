//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
//#include "umpire/DeviceAllocator.hpp"
#include "umpire/util/device_allocator_helper.hpp"
#include "umpire/ResourceManager.hpp"

__global__ void my_kernel()
{
  if (threadIdx.x == 0) {
    umpire::DeviceAllocator alloc = umpire::util::getDeviceAllocator(0);

    double* data = static_cast<double*>(alloc.allocate(1 * sizeof(double)));
    data[0] = 1024.5;
  }
}

int main(int argc, char const* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = rm.makeDeviceAllocator(allocator, 1024);
  auto device_allocator2 = rm.makeDeviceAllocator(allocator, 42);
  auto device_allocator3 = rm.makeDeviceAllocator(allocator, 42);

  rm.syncDeviceAllocator();
  
  my_kernel<<<1, 16>>>();
  cudaDeviceSynchronize();

  //std::cout << device_allocator << std::endl;
  std::cout << "DA1 with ID:" << device_allocator.getID() << std::endl;
  std::cout << "DA2 with ID:" << device_allocator2.getID() << std::endl;
  std::cout << "DA3 with ID:" << device_allocator3.getID() << std::endl;

  return 0;
}
