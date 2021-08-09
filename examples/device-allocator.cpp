//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
//#include "umpire/DeviceAllocator.hpp"
#include "umpire/util/device_allocator_helper.hpp"
#include "umpire/ResourceManager.hpp"

__global__ void my_kernel(double** data_ptr)
{
  if (threadIdx.x == 0) {
    umpire::DeviceAllocator alloc = umpire::util::getDeviceAllocator(0);
    alloc.allocate(10 * sizeof(double));
    //*data_ptr = data;
    //data[7] = 1024;
    //*data_ptr = (double*)alloc->getID();
  }
}

int main(int argc, char const* argv[])
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto device_allocator = rm.makeDeviceAllocator(allocator, 1024);


std::cout << "size of DA: " << sizeof(device_allocator) << " compared to global var's:"
          << sizeof(umpire::util::UMPIRE_DEV_ALLOCS_h[0]) << std::endl;
  auto my_alloc = umpire::util::UMPIRE_DEV_ALLOCS_h[0];
std::cout << "global var has: " << my_alloc.getID() <<std::endl;

//  rm.syncDeviceAllocator();
//  double** ptr_to_data =
//      static_cast<double**>(allocator.allocate(sizeof(double*)));


  //my_kernel<<<1, 16>>>(ptr_to_data);
  //cudaDeviceSynchronize();

  //std::cout << (*ptr_to_data)[7] << std::endl;
  //std::cout << "DA1 with ID:" << device_allocator.getID() << std::endl;

  return 0;
}
