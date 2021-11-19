//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HipCopyToOperation.hpp"

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void HipCopyToOperation::transform(void* src_ptr, void** dst_ptr,
                                   umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                                   umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
                                   std::size_t length)
{
  hipError_t error = ::hipMemcpy(*dst_ptr, src_ptr, length, hipMemcpyHostToDevice);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,"hipMemcpy( dest_ptr = " << *dst_ptr << ", src_ptr = " << src_ptr << ", length = " << length
                                          << ", hipMemcpyHostToDevice ) failed with error: "
                                          << hipGetErrorString(error));
  }
}

camp::resources::EventProxy<camp::resources::Resource> HipCopyToOperation::transform_async(
    void* src_ptr, void** dst_ptr, umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length,
    camp::resources::Resource& ctx)
{
  auto device = ctx.get<camp::resources::Hip>();
  auto stream = device.get_stream();

  hipError_t error = ::hipMemcpyAsync(*dst_ptr, src_ptr, length, hipMemcpyHostToDevice, stream);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,"hipMemcpyAsync( dest_ptr = "
                 << *dst_ptr << ", src_ptr = " << src_ptr << ", length = " << length << ", hipMemcpyHostToDevice "
                 << ", stream = " << stream << ") failed with error: " << hipGetErrorString(error));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
