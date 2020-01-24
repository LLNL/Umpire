//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/OpenMPTargetCopyOperation.hpp"

#include <cstring>

#include "umpire/util/Macros.hpp"

#include "omp.h"

namespace umpire {
namespace op {

OpenMPTargetCopyOperation::OpenMPTargetCopyOperation(int src, int dst) :
  m_src_id{src},
  m_dst_id{dst}
{}

void OpenMPTargetCopyOperation::transform(
    void* src_ptr,
    void** dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    std::size_t length)
{
  UMPIRE_LOG(Debug, "omp_target_memcpy( dst_ptr = " << *dst_ptr
      << ", src_ptr = " << src_ptr
      << ", length = " << length
      << ", src_id = " << m_src_id
      << ", dst_id = " << m_dst_id);

  omp_target_memcpy( 
      *dst_ptr,
      src_ptr, 
      length,
      0, 0, 
      m_dst_id,
      m_src_id); 

  UMPIRE_RECORD_STATISTIC(
      "OpenMPTargetCopyOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(dst_ptr),
      "size", length,
      "event", "copy");
}

} // end of namespace op
} // end of namespace umpire
