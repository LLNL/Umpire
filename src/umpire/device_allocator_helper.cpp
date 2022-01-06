////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "umpire/device_allocator_helper.hpp"

#include <limits.h>
#include <string.h>

#include <functional>

#include "umpire/ResourceManager.hpp"
#include "umpire/alloc/CudaMallocManagedAllocator.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {

//////////////////////////////////////////////////////////////////////////
// Global variables for host and device
//////////////////////////////////////////////////////////////////////////
DeviceAllocator* UMPIRE_DEV_ALLOCS_h{nullptr};
__device__ DeviceAllocator* UMPIRE_DEV_ALLOCS{nullptr};

//////////////////////////////////////////////////////////////////////////
// host/device functions
//////////////////////////////////////////////////////////////////////////
__host__ __device__ DeviceAllocator get_device_allocator(const char* name)
{
  int index{-1};
#if !defined(__CUDA_ARCH__)
  for (int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS; i++) {
    if (strcmp(UMPIRE_DEV_ALLOCS_h[i].getName(), name) == 0) {
      index = i;
    }
  }
#else
  for (int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS; i++) {
    const char* temp = UMPIRE_DEV_ALLOCS[i].getName();
    int curr = 0;
    int tally = 0;
    do {
      if (temp[curr] == 0) {
        break;
      }
      if (temp[curr] != name[curr]) {
        tally++;
      }
    } while (name[curr++] != 0);
    if (tally == 0) {
      index = i;
      break;
    }
  }
#endif

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
  if (id < 1 || id > UMPIRE_TOTAL_DEV_ALLOCS) {
    UMPIRE_ERROR("Invalid ID given. Valid range: [1-64]");
  }
  if (!is_device_allocator(id)) {
    UMPIRE_ERROR("No DeviceAllocator by with that ID was found.");
  }

  id = id - 1; // adjust for indexing into array

#if !defined(__CUDA_ARCH__)
  return UMPIRE_DEV_ALLOCS_h[id];
#else
  return UMPIRE_DEV_ALLOCS[id];
#endif
}

__host__ __device__ bool is_device_allocator(int id)
{
  if (id < 1 || id > UMPIRE_TOTAL_DEV_ALLOCS) {
#if !defined(__CUDA_ARCH__)
    UMPIRE_LOG(Warning, "Invalid ID given. Valid range: [1-64]");
    return false;
#else
    UMPIRE_ERROR("Invalid ID given. Valid range: [1-64]");
#endif
  }

  id = id - 1; // adjust for indexing into array

#if !defined(__CUDA_ARCH__)
  return UMPIRE_DEV_ALLOCS_h[id].isInitialized();
#else
  return UMPIRE_DEV_ALLOCS[id].isInitialized();
#endif
}

//////////////////////////////////////////////////////////////////////////
// host functions
//////////////////////////////////////////////////////////////////////////
__host__ DeviceAllocator make_device_allocator(Allocator allocator, size_t size, const std::string& name)
{
  static int allocator_id{0};
  int da_id = ((-1) * (allocator_id)) - 1;

  // The DA ID should not conflict with other allocator IDs,
  // so we use negative numbers to get unique value.
  auto dev_alloc = DeviceAllocator(allocator, size, name, da_id);

  if (UMPIRE_DEV_ALLOCS_h == nullptr) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto um_alloc = rm.getAllocator("UM");
    UMPIRE_DEV_ALLOCS_h =
        (umpire::DeviceAllocator*)um_alloc.allocate(UMPIRE_TOTAL_DEV_ALLOCS * sizeof(DeviceAllocator));
  }

  UMPIRE_DEV_ALLOCS_h[allocator_id++] = dev_alloc;

  // Call macro so that host and device pointers are set up correctly
  UMPIRE_SET_UP_DEVICE_ALLOCATORS();

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
