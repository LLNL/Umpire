////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "umpire/util/device_allocator_helper.hpp"

namespace umpire {

namespace util {

DeviceAllocator* UMPIRE_DEV_ALLOCS_h{nullptr};
__device__ DeviceAllocator* UMPIRE_DEV_ALLOCS{nullptr};

} // end of namespace util

} // end of namespace umpire
