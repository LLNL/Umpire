//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaMemPrefetchOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

void CudaMemPrefetchOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value,
                                     std::size_t length)
{
  int device{value};
  cudaError_t error;

  // Use current device for properties if device is CPU
  int current_device;
  cudaGetDevice(&current_device);
  int gpu = (device != cudaCpuDeviceId) ? device : current_device;

  cudaDeviceProp properties;
  error = ::cudaGetDeviceProperties(&properties, gpu);

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("cudaGetDeviceProperties( device = {} ) failed with error: {}",
                                            device, cudaGetErrorString(error)));
  }

  if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
    error = ::cudaMemPrefetchAsync(src_ptr, length, device);

    if (error != cudaSuccess) {
      UMPIRE_ERROR(
          runtime_error,
          fmt::format("cudaMemPrefetchAsync( src_ptr = {}, length = {}, device = {}) failed with error: {}",
                      src_ptr, length, device, cudaGetErrorString(error)));
    }
  }
}

camp::resources::EventProxy<camp::resources::Resource> CudaMemPrefetchOperation::apply_async(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation), int value, std::size_t length,
    camp::resources::Resource& ctx)
{
  int device{value};
  cudaError_t error;

  // Use current device for properties if device is CPU
  int current_device;
  cudaGetDevice(&current_device);
  int gpu = (device != cudaCpuDeviceId) ? device : current_device;

  cudaDeviceProp properties;
  error = ::cudaGetDeviceProperties(&properties, gpu);

  auto resource = ctx.try_get<camp::resources::Cuda>();
  if (!resource) {
    UMPIRE_ERROR(resource_error, fmt::format("Expected resources::Cuda, got resources::{}",
                                             platform_to_string(ctx.get_platform())));
  }
  auto stream = resource->get_stream();

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("cudaGetDeviceProperties( device = {} ) failed with error: {}",
                                            device, cudaGetErrorString(error)));
  }

  if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
    error = ::cudaMemPrefetchAsync(src_ptr, length, device, stream);

    if (error != cudaSuccess) {
      UMPIRE_ERROR(
          runtime_error,
          fmt::format(
              "cudaMemPrefetchAsync( src_ptr = {}, length = {}, device = {}, stream = {}) failed with error: {}",
              src_ptr, length, device, cudaGetErrorString(error), (void*)stream));
    }
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
