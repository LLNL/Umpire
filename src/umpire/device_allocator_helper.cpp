////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "umpire/device_allocator_helper.hpp"

#include <string.h>

#include "umpire/ResourceManager.hpp"
#include "umpire/alloc/CudaMallocManagedAllocator.hpp"

namespace umpire {

//////////////////////////////////////////////////////////////////////////
// Global variables for host and device
//////////////////////////////////////////////////////////////////////////
__managed__ int UMPIRE_TOTAL_DEV_ALLOCS{10};

DeviceAllocator* UMPIRE_DEV_ALLOCS_h{nullptr};
__device__ DeviceAllocator* UMPIRE_DEV_ALLOCS{nullptr};

//////////////////////////////////////////////////////////////////////////
// host/device functions
//////////////////////////////////////////////////////////////////////////
__host__ __device__ DeviceAllocator get_device_allocator(const char* name)
{
  int index = find_device_allocator_ID(name);
  if (index == -1) {
    UMPIRE_ERROR("No DeviceAllocator by the name " << name << " was found.");
  }
#if !defined(__CUDA_ARCH__)
  return UMPIRE_DEV_ALLOCS_h[index];
#else
  return UMPIRE_DEV_ALLOCS[index];
#endif
}

__host__ __device__ DeviceAllocator get_device_allocator(int id)
{
  if (id < 0 || id > UMPIRE_TOTAL_DEV_ALLOCS) {
    UMPIRE_ERROR("Invalid ID given.");
  }
  if (!is_device_allocator(id)) {
    UMPIRE_ERROR("No DeviceAllocator by with that ID was found.");
  }

#if !defined(__CUDA_ARCH__)
  return UMPIRE_DEV_ALLOCS_h[id];
#else
  return UMPIRE_DEV_ALLOCS[id];
#endif
}

__host__ __device__ int find_device_allocator_ID(const char* name)
{
#if !defined(__CUDA_ARCH__)
  for (int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS; i++) {
    if (strcmp(UMPIRE_DEV_ALLOCS_h[i].getName(), name) == 0) {
      return i;
    }
  }
  return -1;
#else
  for (int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS; i++) {
    const char* temp = UMPIRE_DEV_ALLOCS[i].getName();
    int index = 0;
    int tally = 0;
    do {
      if (temp[index] == 0) {
        break;
      }
      if (temp[index] != name[index]) {
        tally++;
      }
    } while (name[index++] != 0);
    if (tally == 0) {
      return i;
    }
  }
  return -1;
#endif
}

__host__ __device__ bool is_device_allocator(int id)
{
  if (id < 0 || id > UMPIRE_TOTAL_DEV_ALLOCS) {
    UMPIRE_ERROR("Invalid ID given.");
  }

#if !defined(__CUDA_ARCH__)
  return UMPIRE_DEV_ALLOCS_h[id].isInitialized();
#else
  return UMPIRE_DEV_ALLOCS[id].isInitialized();
#endif
}

//////////////////////////////////////////////////////////////////////////
// host functions
//////////////////////////////////////////////////////////////////////////
__host__ DeviceAllocator make_device_allocator(Allocator allocator, size_t size, const char* name)
{
  static size_t allocator_id{0};
  auto dev_alloc = DeviceAllocator(allocator, size, name, allocator_id);

  if (allocator_id == 0) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto um_alloc = rm.getAllocator("UM");
    UMPIRE_DEV_ALLOCS_h =
        (umpire::DeviceAllocator*)um_alloc.allocate(UMPIRE_TOTAL_DEV_ALLOCS * sizeof(DeviceAllocator));
  }

  UMPIRE_DEV_ALLOCS_h[allocator_id++] = dev_alloc;
  return dev_alloc;
}

__host__ void destroy_device_allocator()
{
  for (int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS; i++) {
    if (UMPIRE_DEV_ALLOCS_h[i].isInitialized()) {
      UMPIRE_DEV_ALLOCS_h[i].destroy();
    }
  }
}

} // end of namespace umpire
