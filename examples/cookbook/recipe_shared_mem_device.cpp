//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/strategy/NamingShim.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

__global__ void touch_data(double* data, int len)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < len) {
    data[id] = id * 1024;
  }
}

int main(int, char**)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto traits{umpire::get_default_resource_traits("SHARED")};
  traits.size = 10 * 1024 * 1024;                                   // Maximum size of this Allocator
  traits.scope = umpire::MemoryResourceTraits::shared_scope::node; // default
  auto node_allocator{rm.makeResource("SHARED::node_allocator", traits)};
  auto shim{rm.makeAllocator<umpire::strategy::NamingShim>("shim", node_allocator)};

  double* ptr = static_cast<double*>(shim.allocate(4096*sizeof(double)));
  hipError_t err = hipHostRegister(ptr, 4096*sizeof(double), hipHostRegisterDefault);
  if (err != hipSuccess) {
    std::cerr << "Error registering host memory: " << err << std::endl;
  }

  touch_data<<<16, 256>>>(ptr, 4096);

  hipDeviceSynchronize();

  for (int i = 0; i < 4096; i+=256) {
    std::cout << "Ptr[1] = " << ptr[i] << std::endl;
  }

  std::cout << "Total Memory Allocated: " << umpire::get_total_bytes_allocated() << std::endl;
  shim.deallocate(ptr);

  return 0;
}
