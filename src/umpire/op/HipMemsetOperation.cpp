//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HipMemsetOperation.hpp"

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void HipMemsetOperation::apply(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation),
    int value, std::size_t length)
{
  hipError_t error = ::hipMemset(src_ptr, value, length);

  if (error != hipSuccess) {
    UMPIRE_ERROR("hipMemset( src_ptr = "
                 << src_ptr << ", value = " << value << ", length = " << length
                 << ") failed with error: " << hipGetErrorString(error));
  }
}

} // end of namespace op
} // end of namespace umpire
