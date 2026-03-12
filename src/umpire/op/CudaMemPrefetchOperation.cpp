//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
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
  error = cudaGetDevice(&current_device);
  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("cudaGetDevice failed with error: {}", cudaGetErrorString(error)));
  }
  int gpu = (device != cudaCpuDeviceId) ? device : current_device;
#if CUDART_VERSION >= 13000
  cudaMemLocation loc = {(device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice, device};
#endif

  cudaDeviceProp properties;
  error = ::cudaGetDeviceProperties(&properties, gpu);
  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("cudaGetDeviceProperties( device = {} ) failed with error: {}", gpu,
                                            cudaGetErrorString(error)));
  }

  if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
#if CUDART_VERSION >= 13000
    error = ::cudaMemPrefetchAsync(src_ptr, length, loc, 0);
#else
    error = ::cudaMemPrefetchAsync(src_ptr, length, device);
#endif

    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
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
  error = cudaGetDevice(&current_device);
  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("cudaGetDevice failed with error: {}", cudaGetErrorString(error)));
  }
  int gpu = (device != cudaCpuDeviceId) ? device : current_device;
#if CUDART_VERSION >= 13000
  cudaMemLocation loc = {(device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice, device};
#endif

  cudaDeviceProp properties;
  error = ::cudaGetDeviceProperties(&properties, gpu);
  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("cudaGetDeviceProperties( device = {} ) failed with error: {}", gpu,
                                            cudaGetErrorString(error)));
  }

  auto resource = ctx.try_get<camp::resources::Cuda>();
  if (!resource) {
    UMPIRE_ERROR(resource_error,
                 fmt::format("Expected resources::Cuda, got resources::{}", platform_to_string(ctx.get_platform())));
  }
  auto stream = resource->get_stream();

  if (properties.managedMemory == 1 && properties.concurrentManagedAccess == 1) {
#if CUDART_VERSION >= 13000
    error = ::cudaMemPrefetchAsync(src_ptr, length, loc, 0, stream);
#else
    error = ::cudaMemPrefetchAsync(src_ptr, length, device, stream);
#endif

    if (error != cudaSuccess) {
      UMPIRE_ERROR(
          runtime_error,
          fmt::format(
              "cudaMemPrefetchAsync( src_ptr = {}, length = {}, device = {}, stream = {}) failed with error: {}",
              src_ptr, length, device, (void*)stream, cudaGetErrorString(error)));
    }
  }

  return camp::resources::EventProxy<camp::resources::Resource>{ctx};
}

} // end of namespace op
} // end of namespace umpire
