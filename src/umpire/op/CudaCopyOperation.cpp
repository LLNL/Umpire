//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaCopyOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

void CudaCopyOperation::transform(void* src_ptr, void** dst_ptr,
                                  umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                                  umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length)
{
  cudaError_t error = ::cudaMemcpy(*dst_ptr, src_ptr, length, cudaMemcpyDeviceToDevice);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(
        runtime_error,
        umpire::fmt::format(
            "CudaMemmcpy( dest_ptr = {}, src_ptr = {}, length = {}, cudaMemcpyDeviceToDevice) failed with error: {}",
            *dst_ptr, src_ptr, length, cudaGetErrorString(error)));
  }
}

camp::resources::EventProxy<camp::resources::Resource> CudaCopyOperation::transform_async(
    void* src_ptr, void** dst_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length, camp::resources::Resource& ctx)
{
  auto device = ctx.get<camp::resources::Cuda>();
  auto stream = device.get_stream();

  cudaError_t error = ::cudaMemcpyAsync(*dst_ptr, src_ptr, length, cudaMemcpyDeviceToDevice, stream);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("cudaMemcpyAsync( dest_ptr = {}, src_ptr = {}, length = {}, "
                                     "cudaMemcpyDeviceToDevice, stream = {}) failed with error: {}",
                                     *dst_ptr, src_ptr, length, cudaGetErrorString(error), (void*)stream));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
