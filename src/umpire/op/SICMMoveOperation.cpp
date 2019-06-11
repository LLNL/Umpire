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

sicm_device_list SICMMoveOperation::m_devices = sicm_init();

SICMMoveOperation::~SICMMoveOperation() {
    sicm_fini();
}

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
              static_cast<strategy::AllocationTracker *>(dst_allocation->m_strategy)->getAllocationStrategy()));
  {
    sicm_arena sa = sicm_arena_lookup(src_ptr);

    if (sicm_arena_set_device(sa, &m_devices.devices[dst_allocator->getDeviceIndex()]) != 0) {
        const int err = errno;
        sicm_fini();
        UMPIRE_ERROR("SICMMoveOperation error: " << strerror(err));
    }

    *dst_ptr = src_ptr;

    // find the arena the src_ptr was at and move the record to the new device
    {
      std::lock_guard <std::mutex> lock(alloc::SICMAllocator::arena_mutex);
      bool found = false;
      for(std::pair <const unsigned int, std::list <sicm_arena> > & device : alloc::SICMAllocator::arenas) {
        for(std::list <sicm_arena>::const_iterator it = device.second.begin(); it != device.second.end(); it++) {
          if (*it == sa) {
            device.second.erase(it);
            found = true;
            break;
          }
        }

        if (found) {
          break;
        }
      }

      if (found) {
        alloc::SICMAllocator::arenas[dst_allocator->getDeviceIndex()].push_back(sa);
      }
      else {
          // is this an error?
      }
    }
  }

  (void) length; // length is not detected as being used by the macro

  UMPIRE_RECORD_STATISTIC(
      "SICMMoveOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(dst_ptr),
      "size", length,
      "event", "move");
}

} // end of namespace op
} // end of namespace umpire
