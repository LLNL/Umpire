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
 */
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
__global__ void umpire_device_memset_kernel(void* data, std::size_t count, int value)
{
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;

  for (std::size_t i = idx; i < count; i += stride) {
    data[i] = value;
  }
}
#endif

/*!
 * \brief Launch a device kernel to set DEVICE memory to a value.
 *
 * \param ptr Void pointer to DEVICE memory.
 * \param count Number of elements.
 * \param value Value to assign to each element (may be 0, -1, NaN, etc.).
 */
void device_memset_kernel_impl(void* ptr, std::size_t count, int value);
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
  umpire_device_memset_kernel<<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
      ptr, count, value);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("device_memset kernel launch failed for ptr = {}, count = {} with error: {}",
                             static_cast<void*>(ptr), count, cudaGetErrorString(err)));
  }
#elif defined(__HIP_DEVICE_COMPILE__)
  hipLaunchKernelGGL(umpire_device_memset_kernel, dim3(static_cast<unsigned int>(grid_size)),
                     dim3(block_size), 0, stream, ptr, count, value);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("device_memset kernel launch failed for ptr = {}, count = {} with error: {}",
                             static_cast<void*>(ptr), count, hipGetErrorString(err)));
  }
#endif
}

void device_memset_kernel_nan_impl(static_cast<void*>(ptr), n * sizeof(T))
{


}

} // end of namespace umpire

#endif // UMPIRE_device_memset_kernel_HPP
