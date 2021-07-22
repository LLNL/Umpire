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

void HipCopyOperation::transform(void* src_ptr, void** dst_ptr,
                                 umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                                 umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length)
{
  hipError_t error = ::hipMemcpy(*dst_ptr, src_ptr, length, hipMemcpyDeviceToDevice);

  if (error != hipSuccess) {
    UMPIRE_ERROR("hipMemcpy( dest_ptr = " << *dst_ptr << ", src_ptr = " << src_ptr << ", length = " << length
                                          << ", hipMemcpyDeviceToDevice ) failed with error: "
                                          << hipGetErrorString(error));
  }
}

camp::resources::EventProxy<camp::resources::Resource> HipCopyOperation::transform_async(void* src_ptr, void** dst_ptr,
                                                         util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                                                         util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
                                                         std::size_t length, camp::resources::Resource& ctx)
{
  auto device = ctx.get<camp::resources::Hip>();
  auto stream = device.get_stream();

  hipError_t error = ::hipMemcpyAsync(*dst_ptr, src_ptr, length, hipMemcpyDeviceToDevice, stream);

  if (error != hipSuccess) {
    UMPIRE_ERROR("hipMemcpy( dest_ptr = " << *dst_ptr << ", src_ptr = " << src_ptr << ", length = " << length
                                          << ", hipMemcpyDeviceToHost ) failed with error: "
                                          << hipGetErrorString(error));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
