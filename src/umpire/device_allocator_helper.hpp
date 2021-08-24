////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_device_allocator_helper_HPP
#define UMPIRE_device_allocator_helper_HPP

#include "umpire/DeviceAllocator.hpp"
#include <string.h>

namespace umpire {

extern int UMPIRE_TOTAL_DEV_ALLOCS_h;
extern int UMPIRE_DEV_ALLOCS_COUNTER_h;
extern DeviceAllocator* UMPIRE_DEV_ALLOCS_h;
__device__ extern DeviceAllocator* UMPIRE_DEV_ALLOCS;

__device__ extern DeviceAllocator getDeviceAllocator(const char* name);

__device__ inline DeviceAllocator getDeviceAllocator(int id)
{
  return umpire::UMPIRE_DEV_ALLOCS[id];
}

inline bool existsDeviceAllocator()
{
  return (umpire::UMPIRE_DEV_ALLOCS_h != nullptr) ? true : false;
}

/*!
 * \brief Construct a new DeviceAllocator. Calls the private Device
 * Allocator constructor that records the associated id.
 *
 * \param allocator Allocator to build the DeviceAllocator from.
 * \param size Total size of the DeviceAllocator.
 */
extern DeviceAllocator makeDeviceAllocator(Allocator allocator, size_t size, const char* name);

extern void destroyDeviceAllocator();

#define UMPIRE_SYNC_DEVICE_ALLOCATORS()                       \
{                                                             \
  cudaMemcpyToSymbol(umpire::UMPIRE_DEV_ALLOCS,         \
                    &umpire::UMPIRE_DEV_ALLOCS_h,       \
                    sizeof(umpire::DeviceAllocator*));        \
}

} // end of namespace umpire

#endif // UMPIRE_device_allocator_helper_HPP
