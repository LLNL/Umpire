////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "umpire/device_allocator_helper.hpp"
#include "umpire/ResourceManager.hpp"

#include <string.h>

#include "umpire/alloc/CudaMallocManagedAllocator.hpp"

namespace umpire {

const int UMPIRE_TOTAL_DEV_ALLOCS_h{10};
__device__ const int UMPIRE_TOTAL_DEV_ALLOCS{10};

DeviceAllocator* UMPIRE_DEV_ALLOCS_h{nullptr};
__device__ DeviceAllocator* UMPIRE_DEV_ALLOCS{nullptr};

__device__ DeviceAllocator getDeviceAllocator(const char* name)
{
  for (int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS; i++) {
    const char* temp = UMPIRE_DEV_ALLOCS[i].getName();
    int index = 0;
    int tally = 0;
    do {
      if (temp[index] != name[index])
        tally++;
    } while (name[index++] != 0); 
    if (tally == 0)
      return UMPIRE_DEV_ALLOCS[i];
  }
  UMPIRE_ERROR("No DeviceAllocator by the name " << name << " was found.");
  //
  // The UMPIRE_ERROR macro above does not return.  It instead throws
  // an exception.  However, for some reason, nvcc throws a warning
  // "warning: missing return statement at end of non-void function"
  // even though the following line cannot be reached.  Adding this
  // fake return statement to work around the incorrect warning.
  //
  return UMPIRE_DEV_ALLOCS[0];
}

__device__ DeviceAllocator getDeviceAllocator(int id)
{
  if (id < 0 || id > UMPIRE_TOTAL_DEV_ALLOCS)
    UMPIRE_ERROR("Invalid ID given.");
  if (!deviceAllocatorExistsOnDevice(id))
    UMPIRE_ERROR("No DeviceAllocator by with that ID was found.");
    
  return UMPIRE_DEV_ALLOCS[id];
}

__host__ DeviceAllocator makeDeviceAllocator(Allocator allocator, size_t size, const char* name)
{
  static size_t allocator_id{0};
  auto dev_alloc = DeviceAllocator(allocator, size, name, allocator_id);

  if (allocator_id == 0) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto um_alloc = rm.getAllocator("UM");
    UMPIRE_DEV_ALLOCS_h =
        (umpire::DeviceAllocator*)um_alloc.allocate(UMPIRE_TOTAL_DEV_ALLOCS_h * sizeof(DeviceAllocator));
  }

  UMPIRE_DEV_ALLOCS_h[allocator_id++] = dev_alloc;
  return dev_alloc;
}

__host__ bool deviceAllocatorExists(const char* name)
{
  for (int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS_h; i++) {
    if (strcmp(UMPIRE_DEV_ALLOCS_h[i].getName(), name) == 0)
      return deviceAllocatorExists(i);
  }
  return false;
}

__host__ bool deviceAllocatorExists(int id)
{
  return (UMPIRE_DEV_ALLOCS_h[id].isInitialized()) ? true : false;
}

//__device__ bool deviceAllocatorExistsOnDevice(const char* name)
//{
//
//}

__device__ bool deviceAllocatorExistsOnDevice(int id)
{
  return (UMPIRE_DEV_ALLOCS[id].isInitialized()) ? true : false;
}

__host__ void destroyDeviceAllocator()
{
  for (int i = 0; i < 10; i++) {
    if (UMPIRE_DEV_ALLOCS_h[i].isInitialized()) {
      UMPIRE_DEV_ALLOCS_h[i].destroy();
    }
  }
}

__host__ void synchronizeDeviceAllocator()
{
#if defined(UMPIRE_ENABLE_CUDA)
  cudaDeviceSynchronize();
#elif defined(UMPIRE_ENABLE_HIP)
  hipDeviceSynchronize();
#endif
}

} // end of namespace umpire
