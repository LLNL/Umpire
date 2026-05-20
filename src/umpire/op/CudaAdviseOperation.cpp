//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaAdviseOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

CudaAdviseOperation::CudaAdviseOperation(cudaMemoryAdvise a) : m_advice{a}
{
}

void CudaAdviseOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation), int val,
                                std::size_t length)
{
  int device = val;
  cudaError_t error;

#if CUDART_VERSION >= 13000
  cudaMemLocation loc = {(device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice, device};
  error = ::cudaMemAdvise(src_ptr, length, m_advice, loc);
#else
  error = ::cudaMemAdvise(src_ptr, length, m_advice, device);
#endif

  if (error != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("cudaMemAdvise( src_ptr = {}, length = {}, device = {}) failed with error: {}", src_ptr,
                             length, device, cudaGetErrorString(error)));
  }
}

} // end of namespace op
} // end of namespace umpire
