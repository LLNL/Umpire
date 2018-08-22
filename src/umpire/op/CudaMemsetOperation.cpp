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
#include "umpire/op/CudaMemsetOperation.hpp"

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
CudaMemsetOperation::apply(
    void* src_ptr,
    util::AllocationRecord*  UMPIRE_UNUSED_ARG(allocation),
    int value,
    size_t length)
{
  cudaError_t error = ::cudaMemset(src_ptr, value, length);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaMemset( src_ptr = " << src_ptr
      << ", value = " << value
      << ", length = " << length
      << ") failed with error: "
      << cudaGetErrorString(error));
  }

  UMPIRE_RECORD_STATISTIC(
      "CudaMemsetOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "value", value,
      "size", length,
      "event", "memset");
}

} // end of namespace op
} // end of namespace umpire
