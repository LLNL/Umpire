//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaCopyFromOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

void CudaCopyFromOperation::transform(void* src_ptr, void** dst_ptr, util::AllocationRecord* src_allocation,
                                      util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length)
{
  int device = src_allocation->strategy->getTraits().id;
  int old_device;
  cudaGetDevice(&old_device);
  cudaSetDevice(device);

  cudaError_t error = ::cudaMemcpy(*dst_ptr, src_ptr, length, cudaMemcpyDeviceToHost);

  cudaSetDevice(old_device);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(
        runtime_error,
        umpire::fmt::format(
            "CudaMemmcpy( dest_ptr = {}, src_ptr = {}, length = {}, cudaMemcpyDeviceToHost) failed with error: {}",
            *dst_ptr, src_ptr, length, cudaGetErrorString(error)));
  }
}

camp::resources::EventProxy<camp::resources::Resource> CudaCopyFromOperation::transform_async(
    void* src_ptr, void** dst_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length, camp::resources::Resource& ctx)
{
  auto device = ctx.get<camp::resources::Cuda>();
  auto stream = device.get_stream();

  cudaError_t error = ::cudaMemcpyAsync(*dst_ptr, src_ptr, length, cudaMemcpyDeviceToHost, stream);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 umpire::fmt::format("cudaMemcpyAsync( dest_ptr = {}, src_ptr = {}, length = {}, "
                                     "cudaMemcpyDeviceToHost, stream = {}) failed with error: {}",
                                     *dst_ptr, src_ptr, length, cudaGetErrorString(error), (void*)stream));
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
