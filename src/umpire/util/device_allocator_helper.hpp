////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_device_allocator_helper_HPP
#define UMPIRE_device_allocator_helper_HPP

#include "umpire/DeviceAllocator.hpp"

namespace umpire {

namespace util {

extern DeviceAllocator* UMPIRE_DEV_ALLOCS_h;
__device__ extern DeviceAllocator* UMPIRE_DEV_ALLOCS;

__device__ inline DeviceAllocator getDeviceAllocator(int id)
{
  return umpire::util::UMPIRE_DEV_ALLOCS[id];
}

inline bool existsDeviceAllocator()
{
  return (UMPIRE_DEV_ALLOCS_h != nullptr) ? true : false;
}

#define UMPIRE_SET_UP_DEVICE_ALLOCATOR_ARRAY()           \
{                                                        \
  cudaMemcpyToSymbol(umpire::util::UMPIRE_DEV_ALLOCS,    \
                    &umpire::util::UMPIRE_DEV_ALLOCS_h,  \
                    sizeof(umpire::DeviceAllocator*));   \
}

} // end of namespace util

} // end of namespace umpire

#endif // UMPIRE_device_allocator_helper_HPP
