//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/SICMMoveOperation.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/strategy/SICMStrategy.hpp"
#include "umpire/alloc/SICMAllocator.hpp"

namespace umpire {
namespace op {

void SICMMoveOperation::transform(
    void* src_ptr,
    void** dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord *dst_allocation,
    size_t length)
{
  auto dst_allocator =
      static_cast<strategy::SICMStrategy *>(
          static_cast<strategy::AllocationStrategy*>(
              static_cast<strategy::AllocationTracker *>(dst_allocation->strategy)->getAllocationStrategy()));
  const sicm_arena sa = sicm_arena_lookup(src_ptr);
  sicm_device_list dst_devices = sicm_arena_get_devices(dst_allocator->getArena());
  const int rc = sicm_arena_set_devices(sa, &dst_devices);
  sicm_device_list_free(&dst_devices);

  if (rc != 0) {
    UMPIRE_ERROR("SICMMoveOperation error: " << strerror(-rc));
  }

  *dst_ptr = src_ptr;

  UMPIRE_USE_VAR(length); // length is not detected as being used

  UMPIRE_RECORD_STATISTIC(
      "SICMMoveOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(dst_ptr),
      "size", length,
      "event", "move");
}

} // end of namespace op
} // end of namespace umpire
