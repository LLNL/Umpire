////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "device_memset_kernel.hpp"

namespace umpire {

/*!
 * \brief device kernel to set elements to a value.
 */
template <typename T>
__global__ void umpire_device_memset_kernel(T* data, std::size_t n, T value)
{
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = idx; i < n; i += stride) {
    data[i] = value;
  }
}

template __global__ void umpire_device_memset_kernel<int>(int*, std::size_t, int);
template __global__ void umpire_device_memset_kernel<float>(float*, std::size_t, float);
template __global__ void umpire_device_memset_kernel<double>(double*, std::size_t, double);
template __global__ void umpire_device_memset_kernel<long>(long*, std::size_t, long);
template __global__ void umpire_device_memset_kernel<unsigned long>(unsigned long*, std::size_t, unsigned long);

} // end of namespace umpire
