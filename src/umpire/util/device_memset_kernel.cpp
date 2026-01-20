////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////
#include "device_memset_kernel.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#elif defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace umpire {

/*!
 * \brief device kernel to set elements to a value.
 *
 * T is typically a scalar type (e.g., int, float, double).
 */
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
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
inline void device_memset(Umpire::Allocator alloc, std::size_t n, T value)
{
  if (!alloc || n == 0) {
    return;
  }

  constexpr int block_size = 256;
  std::size_t grid_size = (n + block_size - 1) / block_size;

  // Clamp grid size to avoid launching an excessively large grid.
  const std::size_t max_blocks = 65535;
  if (grid_size > max_blocks) {
    grid_size = max_blocks;
  }

#if defined(__CUDA_ARCH__)
  umpire_device_memset_kernel<T><<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
      ptr, n, value);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("device_memset kernel launch failed for ptr = {}, n = {} with error: {}",
                             static_cast<void*>(ptr), n, cudaGetErrorString(err)));
  }
#elif defined(__HIP_DEVICE_COMPILE__)
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
} // end of namespace umpire

#endif // UMPIRE_device_memset_kernel_HPP
