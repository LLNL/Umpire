//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_CudaPinnedAllocator_HPP
#define UMPIRE_CudaPinnedAllocator_HPP

#include <cuda_runtime_api.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace alloc {

struct CudaPinnedAllocator {
  void* allocate(std::size_t bytes)
  {
    void* ptr = nullptr;
    cudaError_t error = ::cudaMallocHost(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaMallocHost( bytes = " << bytes
                                              << " ) failed with error: "
                                              << cudaGetErrorString(error));
    } else {
      return ptr;
    }
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    cudaError_t error = ::cudaFreeHost(ptr);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("cudaFreeHost( ptr = " << ptr << " ) failed with error: "
                                          << cudaGetErrorString(error));
    }
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CudaPinnedAllocator_HPP
