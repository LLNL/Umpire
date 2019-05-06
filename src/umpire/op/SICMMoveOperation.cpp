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

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/strategy/SICMStrategy.hpp"

#include <numaif.h>

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
    auto dst_allocator = static_cast<strategy::SICMStrategy *>(static_cast<strategy::AllocationStrategy*>(static_cast<strategy::AllocationTracker *>(dst_allocation->m_strategy)->getAllocationStrategy()));

  // delete ResourceManager::getInstance().deregisterAllocation(src_ptr);

  const int rc = sicm_arena_set_device(sicm_arena_lookup(src_ptr), &m_devices.devices[dst_allocator->getDeviceIndex()]);
  if (rc != 0) {
    const int err = errno;
    sicm_fini();
    UMPIRE_ERROR("SICMMoveOperation error: " << rc << " " << strerror(err));
  }

  *dst_ptr = src_ptr;

  memset(*dst_ptr, 0, length);
  int status = -1;
  move_pages(0, 1, dst_ptr, NULL, &status, 0);

  UMPIRE_RECORD_STATISTIC(
      "SICMMoveOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(dst_ptr),
      "size", length,
      "event", "move");

  // ResourceManager::getInstance().registerAllocation(
  //     *dst_ptr,
  //     new util::AllocationRecord{*dst_ptr, length, allocator});
}

} // end of namespace op
} // end of namespace umpire
