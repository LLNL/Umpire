////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_device_memset_kernel_HPP
#define UMPIRE_device_memset_kernel_HPP

#include <string.h>

#include "umpire/util/error.hpp"
#include "umpire/util/Macros.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#elif defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace umpire {

// Forward declare the kernel
//#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
template <typename T>
__global__ void umpire_device_memset_kernel(T* data, std::size_t n, T value);
#endif

/*!
 * \brief Launch a device kernel to set DEVICE memory to a value.
 *
 * \tparam T Type for the ptr and value used in the memset.
 * \param ptr Pointer to the memory to memset.
 * \param n Number of elements.
 * \param value Value to assign to each element (may be 0, -1, NaN, etc.).
 */
template <typename T>
void device_memset(T* ptr, std::size_t n, T value)
{
  if (!ptr || n == 0) {
    return;
  }

  constexpr int block_size = 256;
  std::size_t grid_size = (n + block_size - 1) / block_size;

  const std::size_t max_blocks = 65535;
  if (grid_size > max_blocks) {
    grid_size = max_blocks;
  }

#if defined(UMPIRE_ENABLE_CUDA)
  umpire_device_memset_kernel<<<static_cast<unsigned int>(grid_size), block_size>>>(ptr, n, value);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("device_memset kernel launch failed: {}", cudaGetErrorString(err)));
  }
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(umpire_device_memset_kernel, dim3(grid_size), dim3(block_size), 0, 0, ptr, n, value);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("device_memset kernel launch failed: {}", hipGetErrorString(err)));
  }
#endif
}


template <typename T>
void device_memset_kernel_nan(T* ptr, std::size_t n) {
}

} // end of namespace umpire

#endif // UMPIRE_device_memset_kernel_HPP
