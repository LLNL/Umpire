//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/device_allocator_helper.hpp"
#include "umpire/ResourceManager.hpp"

__global__ void my_kernel(double** data_ptr)
{
  if (threadIdx.x == 0) {
    umpire::DeviceAllocator alloc = umpire::util::getDeviceAllocator(1);
    double* data = static_cast<double*>(alloc.allocate(10 * sizeof(double)));
    *data_ptr = data;
    data[7] = 1024;

    //*data_ptr = alloc.getBytesUsed();
  }
}

int main(int argc, char const* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = rm.makeDeviceAllocator(allocator, 1024);
  auto device_allocator2 = rm.makeDeviceAllocator(allocator, 2048);

  double** ptr_to_data =
      static_cast<double**>(allocator.allocate(sizeof(double*)));

  std::cout << "Bytes Used: " << device_allocator.getBytesUsed() << std::endl;

  UMPIRE_SET_UP_DEVICE_ALLOCATOR_ARRAY();
  my_kernel<<<1, 16>>>(ptr_to_data);
  
  if (cudaSuccess != cudaGetLastError())
    std::cout<<"kernel ERROR!"<<std::endl;
  
  cudaDeviceSynchronize();

  std::cout << "Bytes Used: " << device_allocator.getBytesUsed() << std::endl;
  //std::cout << "Device reported bytes used: " << ptr_to_data[0] << std::endl;
  std::cout << (*ptr_to_data)[7] << std::endl;
  std::cout << "DA1 with ID:" << device_allocator.getID() << std::endl;
  std::cout << "DA2 with ID:" << device_allocator2.getID() << std::endl;

  return 0;
}
