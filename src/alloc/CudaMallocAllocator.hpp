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
#ifndef UMPIRE_CudaMallocAllocator_HPP
#define UMPIRE_CudaMallocAllocator_HPP

#include <cuda_runtime_api.h>

namespace umpire {
namespace alloc {

/*!
 * \brief Uses cudaMalloc and cudaFree to allocate and deallocate memory on
 *        NVIDIA GPUs.
 */
struct CudaMallocAllocator
  /*!
   * \brief Allocate bytes of memory using cudaMalloc
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(size_t bytes)
  {
    void* ptr = nullptr;
    cudaError_t error = ::cudaMalloc(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaMalloc( bytes = " << bytes << " ) failed with error: " << cudaGetErrorString(error));
    } else {
      return ptr;
    }
  }

  /*!
   * \brief Deallocate memory using cudaFree.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::Exception if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    cudaError_t error = ::cudaFree(ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaFree( ptr = " << ptr << " ) failed with error: " << cudaGetErrorString(error));
    }
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaMallocAllocator_HPP
