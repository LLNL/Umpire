////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "umpire/util/device_allocator_helper.hpp"
#include <string.h>

namespace umpire {

namespace util {

int UMPIRE_TOTAL_DEV_ALLOCS_h{10};
int UMPIRE_DEV_ALLOCS_COUNTER_h{0};
DeviceAllocator* UMPIRE_DEV_ALLOCS_h{nullptr};
__device__ DeviceAllocator* UMPIRE_DEV_ALLOCS{nullptr};

__device__ DeviceAllocator getDeviceAllocator(const char* name)
{
  for (int i = 0; i < 10; i++)
  { 
    if(umpire::util::UMPIRE_DEV_ALLOCS[i].getName() == name) {
      return umpire::util::UMPIRE_DEV_ALLOCS[i];
    }
  }
  return umpire::util::UMPIRE_DEV_ALLOCS[0];
}

} // end of namespace util

} // end of namespace umpire
