//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaAdviseUnsetAccessedByOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void CudaAdviseUnsetAccessedByOperation::apply(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    int val, std::size_t length)
{
  int device = val;
  cudaError_t error;

  cudaDeviceProp properties;
  error = ::cudaGetDeviceProperties(&properties, 0);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaGetDeviceProperties( device = "
                 << 0 << "),"
                 << " failed with error: " << cudaGetErrorString(error));
  }

  if (properties.managedMemory == 1 &&
      properties.concurrentManagedAccess == 1) {
    error =
        ::cudaMemAdvise(src_ptr, length, cudaMemAdviseUnsetAccessedBy, device);

    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaMemAdvise( src_ptr = "
                   << src_ptr << ", length = " << length
                   << ", cudaMemAdviseSetUnsetAccessedBy, " << device
                   << ") failed with error: " << cudaGetErrorString(error));
    }
  }
}

} // end of namespace op
} // end of namespace umpire
