//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaCopyToOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void CudaCopyToOperation::transform(
    void* src_ptr, void** dst_ptr,
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    umpire::util::AllocationRecord* dst_allocation, std::size_t length)
{
  int device = dst_allocation->strategy->getTraits().id;
  int old_device;
  cudaGetDevice(&old_device);
  cudaSetDevice(device);
  cudaError_t error =
      ::cudaMemcpy(*dst_ptr, src_ptr, length, cudaMemcpyHostToDevice);
  cudaSetDevice(old_device);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaMemcpy( dest_ptr = "
                 << *dst_ptr << ", src_ptr = " << src_ptr << ", length = "
                 << length << ", cudaMemcpyHostToDevice ) failed with error: "
                 << cudaGetErrorString(error));
  }

  UMPIRE_RECORD_STATISTIC("CudaCopyToOperation", "src_ptr",
                          reinterpret_cast<uintptr_t>(src_ptr), "dst_ptr",
                          reinterpret_cast<uintptr_t>(dst_ptr), "size", length,
                          "event", "copy");
}

camp::resources::Event CudaCopyToOperation::transform_async(
    void* src_ptr, void** dst_ptr,
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    std::size_t length, camp::resources::Resource& ctx)
{
  auto device = ctx.get<camp::resources::Cuda>();
  auto stream = device.get_stream();

  cudaError_t error = ::cudaMemcpyAsync(*dst_ptr, src_ptr, length,
                                        cudaMemcpyHostToDevice, stream);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaMemcpyAsync( dest_ptr = "
                 << *dst_ptr << ", src_ptr = " << src_ptr
                 << ", length = " << length << ", cudaMemcpyHostToDevice "
                 << ", stream = " << stream
                 << ") failed with error: " << cudaGetErrorString(error));
  }

  UMPIRE_RECORD_STATISTIC("CudaCopyToOperation", "src_ptr",
                          reinterpret_cast<uintptr_t>(src_ptr), "dst_ptr",
                          reinterpret_cast<uintptr_t>(dst_ptr), "size", length,
                          "event", "copy");

  return ctx.get_event();
}

} // end of namespace op
} // end of namespace umpire
