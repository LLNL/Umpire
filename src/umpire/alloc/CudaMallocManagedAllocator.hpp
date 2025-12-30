//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaMallocManagedAllocator_HPP
#define UMPIRE_CudaMallocManagedAllocator_HPP

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses cudaMallocManaged and cudaFree to allocate and deallocate
 *        unified memory on NVIDIA GPUs.
 */
struct CudaMallocManagedAllocator {
  /*!
   * \brief Allocate bytes of memory using cudaMallocManaged.
   *
   * \param bytes Number of bytes to allocate.
   *
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::runtime_error if memory cannot be allocated.
   */
  void* allocate(std::size_t bytes)
  {
    void* ptr = nullptr;
    cudaError_t error = ::cudaMallocManaged(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != cudaSuccess) {
      if (error == cudaErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error, fmt::format("cudaMallocManaged( bytes = {} ) failed with error: {}", bytes,
                                                      cudaGetErrorString(error)));
      } else {
        UMPIRE_ERROR(runtime_error, fmt::format("cudaMallocManaged( bytes = {} ) failed with error: {}", bytes,
                                                cudaGetErrorString(error)));
      }
    }
    return ptr;
  }

  /*!
   * \brief Deallocate memory using cudaFree.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::runtime_error if memory be free'd.
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

  bool isAccessible(Platform p)
  {
    if (p == Platform::cuda || p == Platform::host)
      return true;
    else
      return false; // p is undefined
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocManagedAllocator_HPP
