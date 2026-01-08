//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaMemsetOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

void CudaMemsetOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value,
                                std::size_t length)
{
  cudaError_t error = ::cudaMemset(src_ptr, value, length);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("cudaMemset( src_ptr = {}, val = {}, length = {}) failed with error: {}",
                                            src_ptr, value, length, cudaGetErrorString(error)));
  }
}

camp::resources::EventProxy<camp::resources::Resource> CudaMemsetOperation::apply_async(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value, std::size_t length,
    camp::resources::Resource& ctx)
{
  auto device = ctx.try_get<camp::resources::Cuda>();
  if (!device) {
    UMPIRE_ERROR(resource_error,
                 fmt::format("Expected resources::Cuda, got resources::{}", platform_to_string(ctx.get_platform())));
  }
  auto stream = device->get_stream();

  cudaError_t error = ::cudaMemsetAsync(src_ptr, value, length, stream);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(
        runtime_error,
        fmt::format("cudaMemsetAsync( src_ptr = {}, value = {}, length = {}, stream = {}) failed with error: {}",
                    src_ptr, value, length, cudaGetErrorString(error), (void*)stream));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
