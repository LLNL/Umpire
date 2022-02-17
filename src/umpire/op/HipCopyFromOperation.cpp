//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HipCopyFromOperation.hpp"

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

void HipCopyFromOperation::transform(void* src_ptr, void** dst_ptr,
                                     util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                                     util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length)
{
  hipError_t error = ::hipMemcpy(*dst_ptr, src_ptr, length, hipMemcpyDeviceToHost);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error, umpire::fmt::format("hipMemcpy( dest_ptr = {}, src_ptr = {}, length = {}, hipMemcpyDeviceToHost) failed with error: {}", *dst_ptr, src_ptr, length, hipGetErrorString(error)));
  }
}

camp::resources::EventProxy<camp::resources::Resource> HipCopyFromOperation::transform_async(
    void* src_ptr, void** dst_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length, camp::resources::Resource& ctx)
{
  auto device = ctx.get<camp::resources::Hip>();
  auto stream = device.get_stream();

  hipError_t error = ::hipMemcpyAsync(*dst_ptr, src_ptr, length, hipMemcpyDeviceToHost, stream);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error, umpire::fmt::format("hipMemcpyAsync( dest_ptr = {}, src_ptr = {}, length = {}, hipMemcpyDeviceToHost, stream = {}) failed with error: {}", *dst_ptr, src_ptr, length, hipGetErrorString(error), (void*)stream));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
