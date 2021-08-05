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

int* UMPIRE_DEV_ALLOCS_h;
__device__ int* UMPIRE_DEV_ALLOCS;
__device__ int getDeviceAllocator(int id);

} // end of namespace util

} // end of namespace umpire

#endif // UMPIRE_device_allocator_helper_HPP
