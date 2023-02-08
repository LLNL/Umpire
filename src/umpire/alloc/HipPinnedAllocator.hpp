//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipPinnedAllocator_HPP
#define UMPIRE_HipPinnedAllocator_HPP

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

struct HipPinnedAllocator {
  void* allocate(std::size_t bytes)
  {
    void* ptr{nullptr};

    hipError_t error = ::hipHostMalloc(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != hipSuccess) {
      if (error == hipErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error, umpire::fmt::format("hipMallocHost( bytes = {} ) failed with error: {}",
                                                              bytes, hipGetErrorString(error)));
      } else {
        UMPIRE_ERROR(runtime_error, umpire::fmt::format("hipMallocHost( bytes = {} ) failed with error: {}", bytes,
                                                        hipGetErrorString(error)));
      }
    }

    return ptr;
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    hipError_t error = ::hipHostFree(ptr);
    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                   umpire::fmt::format("hipFreeHost( ptr = {} ) failed with error: {}", ptr, hipGetErrorString(error)));
    }
  }

  bool isAccessible(Platform p)
  {
    if (p == Platform::hip || p == Platform::host)
      return true;
    else
      return false; // p is undefined
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_HipPinnedAllocator_HPP
