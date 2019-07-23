//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HostReallocateOperation.hpp"

#include <cstdlib>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

#include "umpire/strategy/mixins/Inspector.hpp"

namespace umpire {
namespace op {

void HostReallocateOperation::transform(
    void* src_ptr,
    void** dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord *dst_allocation,
    std::size_t length)
{
  auto allocator = dst_allocation->strategy;

  auto old_record = ResourceManager::getInstance().deregisterAllocation(src_ptr);
  *dst_ptr = ::realloc(src_ptr, length);

  if (!*dst_ptr) {
    UMPIRE_ERROR("::realloc(src_ptr=" << src_ptr <<
                 ", old_length=" << old_record.size <<
                 ", length=" << length << ") failed");
  }

  ResourceManager::getInstance().registerAllocation(
     *dst_ptr, {*dst_ptr, length, allocator});

  UMPIRE_RECORD_STATISTIC(
      "HostReallocate",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(*dst_ptr),
      "size", length,
      "event", "reallocate");
}

} // end of namespace op
} // end of namespace umpire
