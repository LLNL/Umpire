//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaMemPrefetchOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
CudaMemPrefetchOperation::apply(
    void* src_ptr,
    util::AllocationRecord*  UMPIRE_UNUSED_ARG(allocation),
    int value,
    std::size_t length)
{
  int device{value};
  cudaError_t error;

  // Use GPU 0 for sync if device is CPU
  int gpu = (device != cudaCpuDeviceId) ? device : 0;

  cudaDeviceProp properties;
  error = ::cudaGetDeviceProperties(&properties, gpu);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaGetDeviceProperties( device = " << device << "),"
        << " failed with error: "
        << cudaGetErrorString(error));
  }

  if (properties.managedMemory == 1
      && properties.concurrentManagedAccess == 1) {
    error = ::cudaMemPrefetchAsync(
        src_ptr, 
        length, 
        device);

    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaMemPrefetchAsync( src_ptr = " << src_ptr
        << ", length = " << length
        << ", device = " << device
        << ") failed with error: "
        << cudaGetErrorString(error));
    }

    // Synchronize the appropriate device
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(gpu);
    cudaDeviceSynchronize();
    cudaSetDevice(current_device);
  }
}

} // end of namespace op
} // end of namespace umpire
