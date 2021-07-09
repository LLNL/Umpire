////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "umpire/DeviceAllocator.hpp"
#include "umpire/util/device_allocator_helper.hpp"

namespace umpire {

namespace util {

void dev_alloc_init()
{
  cudaMallocManaged(&UMPIRE_DEV_ALLOCS, 10*sizeof(DeviceAllocator));
  UMPIRE_DEV_ALLOCS = {0};
}

__device__ DeviceAllocator getDeviceAllocator(size_t id)
{
  return UMPIRE_DEV_ALLOCS[id];
}

} // end namespace util

} // end namespace umpire
