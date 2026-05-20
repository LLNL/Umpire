//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaPinnedAllocator_HPP
#define UMPIRE_CudaPinnedAllocator_HPP

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

struct CudaPinnedAllocator {
  void* allocate(std::size_t bytes)
  {
    void* ptr{nullptr};

    cudaError_t error = ::cudaMallocHost(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != cudaSuccess) {
      if (error == cudaErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error, fmt::format("cudaMallocHost( bytes = {} ) failed with error: {}", bytes,
                                                      cudaGetErrorString(error)));
      } else {
        UMPIRE_ERROR(runtime_error, fmt::format("cudaMallocHost( bytes = {} ) failed with error: {}", bytes,
                                                cudaGetErrorString(error)));
      }
    }

    return ptr;
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    cudaError_t error = ::cudaFreeHost(ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("cudaFreeHost( ptr = {} ) failed with error: {}", ptr, cudaGetErrorString(error)));
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

#endif // UMPIRE_CudaPinnedAllocator_HPP
