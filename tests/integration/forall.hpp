//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_forall_HPP
#define UMPIRE_forall_HPP

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

struct sequential {
};
#if defined(UMPIRE_ENABLE_CUDA)
struct cuda {
};
#endif
#if defined(UMPIRE_ENABLE_HIP)
struct hip {
};
#endif

template <typename LOOP_BODY>
void forall_kernel_cpu(int begin, int end, LOOP_BODY body)
{
  for (int i = 0; i < (end - begin); ++i) {
    body(i);
  }
}

/*
 * \brief Run forall kernel on CPU.
 */
template <typename LOOP_BODY>
void forall(sequential, int begin, int end, LOOP_BODY body)
{
#if defined(UMPIRE_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif
#if defined(UMPIRE_ENABLE_HIP)
  hipDeviceSynchronize();
#endif

  forall_kernel_cpu(begin, end, body);
}

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
template <typename LOOP_BODY>
__global__ void forall_kernel_gpu(int start, int length, LOOP_BODY body)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < length) {
    body(idx);
  }
}
#endif

/*
 * \brief Run forall kernel on GPU.
 */
#if defined(UMPIRE_ENABLE_CUDA)
template <typename LOOP_BODY>
void forall(cuda, int begin, int end, LOOP_BODY&& body)
{
  std::size_t blockSize = 32;
  std::size_t gridSize = (end - begin + blockSize - 1) / blockSize;

  forall_kernel_gpu<<<gridSize, blockSize>>>(begin, end - begin, body);
}
#endif
#if defined(UMPIRE_ENABLE_HIP)
template <typename LOOP_BODY>
void forall(hip, int begin, int end, LOOP_BODY&& body)
{
  std::size_t blockSize = 32;
  std::size_t gridSize = (end - begin + blockSize - 1) / blockSize;

  hipLaunchKernelGGL(forall_kernel_gpu, dim3(gridSize), dim3(blockSize), 0, 0,
                     begin, end - begin, body);
}
#endif

#endif // UMPIRE_forall_HPP
