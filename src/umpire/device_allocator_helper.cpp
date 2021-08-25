////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "umpire/device_allocator_helper.hpp"
#include <string.h>

namespace umpire {

int UMPIRE_TOTAL_DEV_ALLOCS_h{10};
int UMPIRE_DEV_ALLOCS_COUNTER_h{0};
DeviceAllocator* UMPIRE_DEV_ALLOCS_h{nullptr};
__device__ DeviceAllocator* UMPIRE_DEV_ALLOCS{nullptr};

__device__ DeviceAllocator getDeviceAllocator(const char* name)
{
  for (int i = 0; i < 10; i++)
  { 
    if(umpire::UMPIRE_DEV_ALLOCS[i].getName() == name) {
      return umpire::UMPIRE_DEV_ALLOCS[i];
    }
  }
  return umpire::UMPIRE_DEV_ALLOCS[0];
}

DeviceAllocator makeDeviceAllocator(Allocator allocator, size_t size, const char* name)
{
  static size_t i{0};
  auto dev_alloc = DeviceAllocator(allocator, size, name, i);
  umpire::UMPIRE_DEV_ALLOCS_COUNTER_h++;

  if (i == 0) {
    cudaMallocManaged((void**) &umpire::UMPIRE_DEV_ALLOCS_h, 
                                umpire::UMPIRE_TOTAL_DEV_ALLOCS_h*sizeof(DeviceAllocator));
  }

  umpire::UMPIRE_DEV_ALLOCS_h[i++] = dev_alloc;
  return dev_alloc; 
}

bool deviceAllocatorExists(const char* name)
{
  for(int i = 0; i < 10; i++)
  {
    if(umpire::UMPIRE_DEV_ALLOCS_h[i].getName() == name)
      return deviceAllocatorExists(i);
  }  
  return false;
}

bool deviceAllocatorExists(int id)
{
  return (umpire::UMPIRE_DEV_ALLOCS_h[id].isInitialized()) ? true : false;
}

void destroyDeviceAllocator()
{
  for(int i = 0; i < 10; i++) {
    umpire::UMPIRE_DEV_ALLOCS_h[i].destroy();
  }
}

} // end of namespace umpire
