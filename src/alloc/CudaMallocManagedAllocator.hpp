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
#ifndef UMPIRE_CudaMallocManagedAllocator_HPP
#define UMPIRE_CudaMallocManagedAllocator_HPP

#include <cuda_runtime_api.h>

namespace umpire {
namespace alloc {

/*!
 * \brief Uses cudaMallocManaged and cudaFree to allocate and deallocate
 *        unified memory on NVIDIA GPUs.
 */
struct CudaMallocManagedAllocator
{
  /*!
   * \brief Allocate bytes of memory using cudaMallocManaged.
   *
   * \param bytes Number of bytes to allocate.
   *
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(size_t bytes)
  {
    void* ptr = nullptr;
    cudaError_t error = ::cudaMallocManaged(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaMallocManaged( bytes = " << bytes << " ) failed with error: " << cudaGetErrorString(error));
    } else {
      return ptr;
    }
  }

  /*!
   * \brief Deallocate memory using cudaFree.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::Exception if memory be free'd.
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

#endif // UMPIRE_CudaMallocManagedAllocator_HPP
