////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_device_allocator_helper_HPP
#define UMPIRE_device_allocator_helper_HPP

#include <string.h>

#include "umpire/DeviceAllocator.hpp"

namespace umpire {

/*
 * Const variable for the limit of unique DeviceAllocator
 * objects available at once.
 */
extern __managed__ int UMPIRE_TOTAL_DEV_ALLOCS;

/*
 * Global arrays for both host and device which hold the
 * DeviceAllocator objects created.
 */
extern DeviceAllocator* UMPIRE_DEV_ALLOCS_h;
__device__ extern DeviceAllocator* UMPIRE_DEV_ALLOCS;

/*
 * Get the DeviceAllocator object specified by either the given
 * name or id.
 */
__host__ __device__ DeviceAllocator get_device_allocator(const char* name);
__host__ __device__ DeviceAllocator get_device_allocator(int id);

/*
 * Check if the DeviceAllocator object specified by the
 * given id currently exists.
 */
__host__ __device__ bool is_device_allocator(int id);

/*!
 * \brief Construct a new DeviceAllocator. Calls the private Device
 * Allocator constructor that records the associated id.
 *
 * \param allocator Allocator to build the DeviceAllocator from.
 * \param size Total size of the DeviceAllocator.
 */
__host__ DeviceAllocator make_device_allocator(Allocator allocator, size_t size, const std::string name);

/*
 * Destroy any DeviceAllocator objects currently in existence.
 * Deallocate any memory belonging to object about to be destroyed.
 */
__host__ void destroy_device_allocator();

/*
 * This macro ensures that the host and device global arrays
 * that keep track of the DeviceAllocator objects created are
 * synced up and pointing to each other.
 */
#if defined(UMPIRE_ENABLE_CUDA)
#define UMPIRE_SET_UP_DEVICE_ALLOCATORS()                                                                          \
  {                                                                                                                \
    cudaMemcpyToSymbol(umpire::UMPIRE_DEV_ALLOCS, &umpire::UMPIRE_DEV_ALLOCS_h, sizeof(umpire::DeviceAllocator*)); \
  }
#elif defined(UMPIRE_ENABLE_HIP)
#define UMPIRE_SET_UP_DEVICE_ALLOCATORS()                                                                         \
  {                                                                                                               \
    hipMemcpyToSymbol(umpire::UMPIRE_DEV_ALLOCS, &umpire::UMPIRE_DEV_ALLOCS_h, sizeof(umpire::DeviceAllocator*)); \
  }
#else
#define UMPIRE_SET_UP_DEVICE_ALLOCATORS()
#endif

} // end of namespace umpire

#endif // UMPIRE_device_allocator_helper_HPP
