//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/OpenMPTargetCopyOperation.hpp"

#include <cstring>

#include "omp.h"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void OpenMPTargetCopyOperation::transform(
    void* src_ptr, void** dst_ptr, util::AllocationRecord* src_allocation,
    util::AllocationRecord* dst_allocation, std::size_t length)
{
  int src_device = src_allocation->strategy->getTraits().id;
  int dst_device = dst_allocation->strategy->getTraits().id;

  void* src_base_ptr{src_allocation->ptr};
  void* dst_base_ptr{dst_allocation->ptr};

  size_t dst_offset{static_cast<size_t>(static_cast<char*>(*dst_ptr) -
                                        static_cast<char*>(dst_base_ptr))};

  size_t src_offset{static_cast<size_t>(static_cast<char*>(src_ptr) -
                                        static_cast<char*>(src_base_ptr))};

  UMPIRE_LOG(Debug,
             "omp_target_memcpy( dst_ptr = "
                 << dst_base_ptr << ", src_ptr = " << src_base_ptr
                 << ", length = " << length << ", dst_offset = " << dst_offset
                 << ", src_offset = " << src_offset
                 << ", src_id = " << src_device << ", dst_id = " << dst_device);

  omp_target_memcpy(dst_base_ptr, src_base_ptr, length, dst_offset, src_offset,
                    dst_device, src_device);

  UMPIRE_RECORD_STATISTIC("OpenMPTargetCopyOperation", "src_ptr",
                          reinterpret_cast<uintptr_t>(src_ptr), "dst_ptr",
                          reinterpret_cast<uintptr_t>(dst_ptr), "size", length,
                          "event", "copy");
}

} // end of namespace op
} // end of namespace umpire
