//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HipCopyOperation.hpp"

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

HipCopyOperation::HipCopyOperation(hipMemcpyKind kind) : m_kind{kind}
{
}

void HipCopyOperation::transform(void* src_ptr, void** dst_ptr,
                                 umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                                 umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length)
{
  hipError_t error = ::hipMemcpy(*dst_ptr, src_ptr, length, m_kind);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("hipMemcpy( dest_ptr = {}, src_ptr = {}, length = {}) failed with error: {}", *dst_ptr,
                             src_ptr, length, hipGetErrorString(error)));
  }
}

camp::resources::EventProxy<camp::resources::Resource> HipCopyOperation::transform_async(
    void* src_ptr, void** dst_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length, camp::resources::Resource& ctx)
{
  auto device = ctx.try_get<camp::resources::Hip>();
  if (!device) {
    UMPIRE_ERROR(resource_error,
                 fmt::format("Expected resources::Hip, got resources::{}", platform_to_string(ctx.get_platform())));
  }
  auto stream = device->get_stream();

  hipError_t error = ::hipMemcpyAsync(*dst_ptr, src_ptr, length, m_kind, stream);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("hipMemcpyAsync( dest_ptr = {}, src_ptr = {}, length = {}, "
                                            "stream = {}) failed with error: {}",
                                            *dst_ptr, src_ptr, length, (void*)stream, hipGetErrorString(error)));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
