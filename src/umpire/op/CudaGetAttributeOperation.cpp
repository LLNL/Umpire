//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaGetAttributeOperation_INL
#define UMPIRE_CudaGetAttributeOperation_INL

#include "umpire/op/CudaGetAttributeOperation.hpp"

namespace umpire {
namespace op {

template <cudaMemRangeAttribute ATTRIBUTE>
bool CudaGetAttributeOperation<ATTRIBUTE>::check_apply(
    void* src_ptr, umpire::util::AllocationRecord* src_allocation, int val,
    std::size_t length) override
{
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
    int result{-1};

    error = ::cudaMemRangeGetAttribute(&result, sizeof(result), ATTRIBUTE,
                                       src_ptr, length);

    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaMemRangeGetAtribute( src_ptr = "
                   << src_ptr << ", length = " << length
                   << ") failed with error: " << cudaGetErrorString(error));
    }
  }
}

} // end of namespace op
} // end of namespace umpire

#endif // UMPIRE_CudaGetAttributeOperation_HPP
