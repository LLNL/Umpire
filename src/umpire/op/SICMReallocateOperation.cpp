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
#include "umpire/op/SICMReallocateOperation.hpp"

#include <cstdlib>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#include <sicm_low.h>

namespace umpire {
namespace op {

void SICMReallocateOperation::transform(
    void* src_ptr,
    void** dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord *dst_allocation,
    size_t length)
{
  auto allocator = dst_allocation->strategy;

  auto old_record = ResourceManager::getInstance().deregisterAllocation(src_ptr);
  *dst_ptr = sicm_realloc(src_ptr, length);

  if (!*dst_ptr) {
    UMPIRE_ERROR("::realloc(src_ptr=" << src_ptr <<
                 ", old_length=" << old_record.size <<
                 ", length=" << length << ") failed");
  }

  ResourceManager::getInstance().registerAllocation(
     *dst_ptr, {*dst_ptr, length, allocator});

  UMPIRE_RECORD_STATISTIC(
      "SICMReallocate",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(*dst_ptr),
      "size", length,
      "event", "reallocate");
}

} // end of namespace op
} // end of namespace umpire
