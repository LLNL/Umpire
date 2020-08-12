//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HipCopyOperation.hpp"

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void HipCopyOperation::transform(
    void* src_ptr, void** dst_ptr,
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    std::size_t length)
{
  hipError_t error =
      ::hipMemcpy(*dst_ptr, src_ptr, length, hipMemcpyDeviceToDevice);

  if (error != hipSuccess) {
    UMPIRE_ERROR("hipMemcpy( dest_ptr = "
                 << *dst_ptr << ", src_ptr = " << src_ptr << ", length = "
                 << length << ", hipMemcpyDeviceToDevice ) failed with error: "
                 << hipGetErrorString(error));
  }

  UMPIRE_RECORD_STATISTIC("HipCopyOperation", "src_ptr",
                          reinterpret_cast<uintptr_t>(src_ptr), "dst_ptr",
                          reinterpret_cast<uintptr_t>(dst_ptr), "size", length,
                          "event", "copy");
}

} // end of namespace op
} // end of namespace umpire
