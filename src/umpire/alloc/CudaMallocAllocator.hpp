//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaMallocAllocator_HPP
#define UMPIRE_CudaMallocAllocator_HPP

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses cudaMalloc and cudaFree to allocate and deallocate memory on
 *        NVIDIA GPUs.
 */
struct CudaMallocAllocator {
  /*!
   * \brief Allocate bytes of memory using cudaMalloc
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::runtime_error if memory cannot be allocated.
   */
  void* allocate(std::size_t size)
  {
    void* ptr{nullptr};
    cudaError_t error = ::cudaMalloc(&ptr, size);
    UMPIRE_LOG(Debug, "(bytes=" << size << ") returning " << ptr);
    if (error != cudaSuccess) {
      if (error == cudaErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error,
                     fmt::format("cudaMalloc( bytes = {} ) failed with error: {}", size, cudaGetErrorString(error)));
      } else {
        UMPIRE_ERROR(runtime_error,
                     fmt::format("cudaMalloc( bytes = {} ) failed with error: {}", size, cudaGetErrorString(error)));
      }
    }

    return ptr;
  }

  /*!
   * \brief Deallocate memory using cudaFree.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::runtime_error if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    cudaError_t error = ::cudaFree(ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("cudaFree( ptr = {} ) failed with error: {}", ptr, cudaGetErrorString(error)));
    }
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocAllocator_HPP
