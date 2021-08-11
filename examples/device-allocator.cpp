//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
//#include "umpire/DeviceAllocator.hpp"
#include "umpire/util/device_allocator_helper.hpp"
#include "umpire/ResourceManager.hpp"

__global__ void my_kernel(double* data_ptr)
{
  if (threadIdx.x == 0) {
    umpire::DeviceAllocator alloc = umpire::util::getDeviceAllocator(0);
    double* data = static_cast<double*>(alloc.allocate(10 * sizeof(double)));

    //*data_ptr = data;
    //data[0] = 1024;
    *data_ptr = (double)alloc.getID();
  }
}

int main(int argc, char const* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = rm.makeDeviceAllocator(allocator, 1024);


  if (cudaSuccess != rm.syncDeviceAllocator())
    std::cout<<"ERROR!"<<std::endl;
  
  double* ptr_to_data = static_cast<double*>(allocator.allocate(sizeof(double)));

  my_kernel<<<1, 16>>>(ptr_to_data);

  if (cudaSuccess != cudaGetLastError())
    std::cout<<"kernel ERROR!"<<std::endl;
  cudaDeviceSynchronize();

  std::cout << (*ptr_to_data) << std::endl;
  std::cout << "DA1 with ID:" << device_allocator.getID() << std::endl;

  return 0;
}
