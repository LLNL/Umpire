//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/CudaAdvisePreferredLocationOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
CudaAdvisePreferredLocationOperation::apply(
    void* src_ptr,
    util::AllocationRecord* src_allocation,
    int UMPIRE_UNUSED_ARG(val),
    size_t length)
{
  // TODO: get correct device for allocation
  int device = 0;

  if (src_allocation->m_strategy->getPlatform() == cpu) {
    device = cudaCpuDeviceId;
  }

  cudaError_t error =
    ::cudaMemAdvise(src_ptr, length, cudaMemAdviseSetPreferredLocation, device);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaMemAdvise( src_ptr = " << src_ptr
      << ", length = " << length
      << ", cudaMemAdviseSetPreferredLocation, 0) failed with error: "
      << cudaGetErrorString(error));
  }
}

} // end of namespace op
} // end of namespace umpire
