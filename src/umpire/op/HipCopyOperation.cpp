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
#include "umpire/op/HipCopyOperation.hpp"

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void HipCopyOperation::transform(
    void* src_ptr,
    void** dst_ptr,
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    size_t length)
{
  hipError_t error =
    ::hipMemcpy(*dst_ptr, src_ptr, length, hipMemcpyDeviceToDevice);

  if (error != hipSuccess) {
    UMPIRE_ERROR("hipMemcpy( dest_ptr = " << *dst_ptr
      << ", src_ptr = " << src_ptr
      << ", length = " << length
      << ", hipMemcpyDeviceToDevice ) failed with error: "
      << hipGetErrorString(error));
  }

  UMPIRE_RECORD_STATISTIC(
      "HipCopyOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(dst_ptr),
      "size", length,
      "event", "copy");
}

} // end of namespace op
} // end of namespace umpire
