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

extern DeviceAllocator* UMPIRE_DEV_ALLOCS;

void dev_alloc_init();
__device__ DeviceAllocator getDeviceAllocator(size_t id);

} // end of namespace util

} // end of namespace umpire

#endif // UMPIRE_device_allocator_helper_HPP
