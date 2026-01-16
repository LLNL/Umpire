//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC
// and Umpire project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_device_memset_kernel_HPP
#define UMPIRE_device_memset_kernel_HPP

#include <cstddef>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
using stream_type = cudaStream_t;
#elif defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime.h>
using stream_type = hipStream_t;
#endif

namespace umpire {

#if defined (UMPIRE_ENABLE_CUDA) || defined (UMPIRE_ENABLE_HIP)

// Forward declare kernel for host code
template <typename T>
__global__ void umpire_device_memset_kernel(T* data, std::size_t n, T value);

#if defined(__CUDACC__) || defined(__HIPCC__)
/*!
 * \brief device kernel to set elements to a value.
 *
 * T is typically a scalar type (e.g., int, float, double).
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
#endif

/*!
 * \brief Launch a device kernel to set DEVICE memory to a value.
 *
 * \tparam T Element type of the DEVICE allocation.
 * \param ptr Pointer to DEVICE memory (T*).
 * \param n Number of elements.
 * \param value Value to assign to each element (may be 0, -1, NaN, etc.).
 * \param stream CUDA stream to use (defaults to the null stream).
 */
template <typename T>
inline void device_memset(T* ptr, std::size_t n, T value, stream_type stream = 0)
{
  if (!ptr || n == 0) {
    return;
  }

  constexpr int block_size = 256;
  std::size_t grid_size = (n + block_size - 1) / block_size;

  // Clamp grid size to avoid launching an excessively large grid.
  const std::size_t max_blocks = 65535;
  if (grid_size > max_blocks) {
    grid_size = max_blocks;
  }

#if defined(UMPIRE_ENABLE_CUDA)
  umpire_device_memset_kernel<T><<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
      ptr, n, value);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("device_memset kernel launch failed for ptr = {}, n = {} with error: {}",
                             static_cast<void*>(ptr), n, cudaGetErrorString(err)));
  }
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(umpire_device_memset_kernel<T>, dim3(static_cast<unsigned int>(grid_size)),
                     dim3(block_size), 0, stream, ptr, n, value);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("device_memset kernel launch failed for ptr = {}, n = {} with error: {}",
                             static_cast<void*>(ptr), n, hipGetErrorString(err)));
  }
#endif
}

#else

/*!
 * \brief Fallback device_memset for builds without CUDA or HIP.
 *
 * This overload is provided for API completeness but does not perform
 * any GPU work. It will emit a warning when used.
 */
template <typename T>
inline void device_memset(T* UMPIRE_UNUSED_ARG(ptr), std::size_t UMPIRE_UNUSED_ARG(n), T UMPIRE_UNUSED_ARG(value))
{
  UMPIRE_LOG(Warning,
             "device_memset called in a build without CUDA or HIP enabled; no device kernel was launched.");
}

#endif

} // end of namespace umpire

#endif // UMPIRE_device_memset_kernel_HPP
