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
#include "umpire/op/HipMemsetOperation.hpp"

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
HipMemsetOperation::apply(
    void* src_ptr,
    util::AllocationRecord*  UMPIRE_UNUSED_ARG(allocation),
    int value,
    size_t length)
{
  hipError_t error = ::hipMemset(src_ptr, value, length);

  if (error != hipSuccess) {
    UMPIRE_ERROR("hipMemset( src_ptr = " << src_ptr
      << ", value = " << value
      << ", length = " << length
      << ") failed with error: "
      << hipGetErrorString(error));
  }

  UMPIRE_RECORD_STATISTIC(
      "HipMemsetOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "value", value,
      "size", length,
      "event", "memset");
}

} // end of namespace op
} // end of namespace umpire
