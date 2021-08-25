////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "umpire/device_allocator_helper.hpp"
#include "umpire/alloc/CudaMallocManagedAllocator.hpp"
#include <string.h>

namespace umpire {

const int UMPIRE_TOTAL_DEV_ALLOCS_h{10};
__device__ const int UMPIRE_TOTAL_DEV_ALLOCS{10};

DeviceAllocator* UMPIRE_DEV_ALLOCS_h{nullptr};
__device__ DeviceAllocator* UMPIRE_DEV_ALLOCS{nullptr};

__device__ DeviceAllocator getDeviceAllocator(const char* name)
{
  for (int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS; i++)
  { 
    if(UMPIRE_DEV_ALLOCS[i].getName() == name) {
      return UMPIRE_DEV_ALLOCS[i];
    }
  }
  //For now, if we can't find the specified DeviceAllocator,
  // return the first DeviceAllocator.
  return UMPIRE_DEV_ALLOCS[0];
}

__device__ DeviceAllocator getDeviceAllocator(int id)
{
  return UMPIRE_DEV_ALLOCS[id];
}

DeviceAllocator makeDeviceAllocator(Allocator allocator, size_t size, const char* name)
{
  static size_t allocator_id{0};
  auto dev_alloc = DeviceAllocator(allocator, size, name, allocator_id);

  if (allocator_id == 0) {
#if defined (UMPIRE_ENABLE_CUDA)
    auto um_alloc = umpire::alloc::CudaMallocManagedAllocator();
#elif defined (UMPIRE_ENABLE_HIP)
    auto um_alloc = umpire::alloc::HipMallocManagedAllocator();
#endif
    UMPIRE_DEV_ALLOCS_h = (umpire::DeviceAllocator*)um_alloc.allocate(UMPIRE_TOTAL_DEV_ALLOCS_h*sizeof(DeviceAllocator));
  }

  UMPIRE_DEV_ALLOCS_h[allocator_id++] = dev_alloc;
  return dev_alloc; 
}

bool deviceAllocatorExists(const char* name)
{
  for(int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS_h; i++)
  {
    if(UMPIRE_DEV_ALLOCS_h[i].getName() == name)
      return deviceAllocatorExists(i);
  }  
  return false;
}

bool deviceAllocatorExists(int id)
{
  return (UMPIRE_DEV_ALLOCS_h[id].isInitialized()) ? true : false;
}

void destroyDeviceAllocator()
{
  for(int i = 0; i < 10; i++) {
    if(UMPIRE_DEV_ALLOCS_h[i].isInitialized()) {
      UMPIRE_DEV_ALLOCS_h[i].destroy();
    }
  }
}

void synchronizeDeviceAllocator()
{
#if defined (UMPIRE_ENABLE_CUDA)
  cudaDeviceSynchronize();
#elif defined (UMPIRE_ENABLE_HIP)
  hipDeviceSynchronize();
#endif
}

} // end of namespace umpire
