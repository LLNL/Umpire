//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HipMemsetOperation.hpp"

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void HipMemsetOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value,
                               std::size_t length)
{
  hipError_t error = ::hipMemset(src_ptr, value, length);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,"hipMemset( src_ptr = " << src_ptr << ", value = " << value << ", length = " << length
                                         << ") failed with error: " << hipGetErrorString(error));
  }
}

camp::resources::EventProxy<camp::resources::Resource> HipMemsetOperation::apply_async(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value, std::size_t length,
    camp::resources::Resource& ctx)
{
  auto device = ctx.get<camp::resources::Hip>();
  auto stream = device.get_stream();

  hipError_t error = ::hipMemsetAsync(src_ptr, value, length, stream);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,"hipMemset( src_ptr = " << src_ptr << ", value = " << value << ", length = " << length
                                         << ") failed with error: " << hipGetErrorString(error));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
