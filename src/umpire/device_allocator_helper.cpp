////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "umpire/device_allocator_helper.hpp"

#include <string.h>

#include "umpire/ResourceManager.hpp"
#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/alloc/CudaMallocManagedAllocator.hpp"
#else
#include "umpire/alloc/HipMallocManagedAllocator.hpp"
#endif
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
namespace {
/*
 * DeviceAllocator IDs are negative by design so they do not
 * conflict with other allocator IDs. This function converts that
 * negative value to a positive to be used as an array index.
 */
__host__ __device__ inline int convert_to_array_index(int neg_id)
{
  int pos_id = (neg_id * (-1)) - 1;
  return pos_id;
}

/*
 * Given a name, this function returns whether or not it
 * corresponds to a DeviceAllocator.
 */
__host__ __device__ inline int get_index(const char* name)
{
  int index{-1};
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  if (UMPIRE_DEV_ALLOCS_h == nullptr) {
    UMPIRE_LOG(Warning, "No DeviceAllocators have been created yet.");
    return index;
  }

  for (int i = 0; i < UMPIRE_TOTAL_DEV_ALLOCS; i++) {
    if (strcmp(UMPIRE_DEV_ALLOCS_h[i].getName(), name) == 0) {
      index = i;
    }
  }
#else
  if (UMPIRE_DEV_ALLOCS == nullptr) {
    return index;
  }

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

  return index;
}
} // end of namespace

namespace detail {
struct DestroyDeviceAllocatorExit {
  DestroyDeviceAllocatorExit() = default;
  DestroyDeviceAllocatorExit(DestroyDeviceAllocatorExit&&) = delete;
  DestroyDeviceAllocatorExit(const DestroyDeviceAllocatorExit&) = delete;
  DestroyDeviceAllocatorExit& operator=(DestroyDeviceAllocatorExit&&) = delete;
  DestroyDeviceAllocatorExit& operator=(const DestroyDeviceAllocatorExit&) = delete;
  ~DestroyDeviceAllocatorExit()
  {
    if (umpire::UMPIRE_DEV_ALLOCS_h != nullptr) {
      umpire::destroy_device_allocator();
    }
  }
};
} // namespace detail

__host__ __device__ DeviceAllocator get_device_allocator(const char* name)
{
  int index = get_index(name);

  if (index == -1) {
    UMPIRE_ERROR(runtime_error, fmt::format("No DeviceAllocator named \"{}\" was found", name));
  }

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  return UMPIRE_DEV_ALLOCS_h[index];
#else
  return UMPIRE_DEV_ALLOCS[index];
#endif
}

__host__ __device__ DeviceAllocator get_device_allocator(int da_id)
{
  int id = convert_to_array_index(da_id);

  if (id < 0 || id > (UMPIRE_TOTAL_DEV_ALLOCS - 1)) {
    UMPIRE_ERROR(runtime_error, fmt::format("Invalid id given: {}", id));
  }
  if (!is_device_allocator(da_id)) {
    UMPIRE_ERROR(runtime_error, fmt::format("No DeviceAllocator with id: {} was found", id));
  }

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  return UMPIRE_DEV_ALLOCS_h[id];
#else
  return UMPIRE_DEV_ALLOCS[id];
#endif
}

__host__ __device__ bool is_device_allocator(const char* name)
{
  int index = get_index(name);

  if (index == -1) {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    UMPIRE_LOG(Warning, "No DeviceAllocator by the name " << name << " was found.");
    return false;
#else
    UMPIRE_ERROR(runtime_error, fmt::format("No DeviceAllocator by the name \"{}\" was found", name));
#endif
  }

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  return UMPIRE_DEV_ALLOCS_h[index].isInitialized();
#else
  return UMPIRE_DEV_ALLOCS[index].isInitialized();
#endif
}

__host__ __device__ bool is_device_allocator(int da_id)
{
  int id = convert_to_array_index(da_id);

  if (id < 0 || id > (UMPIRE_TOTAL_DEV_ALLOCS - 1)) {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    UMPIRE_LOG(Warning, "Invalid ID given: " << id);
    return false;
#else
    UMPIRE_ERROR(runtime_error, fmt::format("Invalid id given: {}", id));
#endif
  }

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
  static int index{0};

  if (size <= 0) {
    UMPIRE_ERROR(runtime_error, fmt::format("Invalid size passed to DeviceAllocator: ", size));
  }

  if (UMPIRE_DEV_ALLOCS_h == nullptr) {
    index = 0; // If destroy_device_allocator has been called, reset counter.

    auto& rm = umpire::ResourceManager::getInstance();

    // This function-local static will be constructed after the function-local
    // static ResourceManager is constructed, guaranteeing that the destructor
    // of destroy_exit will be called before the destructor of ResourceManager.
    // The DestroyDeviceAllocatorExit destructor releases all allocated
    // DeviceAllocators, unless destroy_device_allocator has already been called
    // manually.
    static detail::DestroyDeviceAllocatorExit destroy_exit;

    auto um_alloc = rm.getAllocator("UM");
    UMPIRE_DEV_ALLOCS_h =
        (umpire::DeviceAllocator*)um_alloc.allocate(UMPIRE_TOTAL_DEV_ALLOCS * sizeof(DeviceAllocator));
  }

  // The DeviceAllocator ID should not conflict with other allocator IDs,
  // so we use negative numbers to get unique value.
  int da_id = convert_to_array_index(index);

  auto dev_alloc = DeviceAllocator(allocator, size, name, da_id);

  UMPIRE_DEV_ALLOCS_h[index++] = dev_alloc;

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
  UMPIRE_DEV_ALLOCS_h = nullptr;
}

} // end of namespace umpire
