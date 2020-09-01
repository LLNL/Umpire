//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipPinnedAllocator_HPP
#define UMPIRE_HipPinnedAllocator_HPP

#include <hip/hip_runtime.h>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace alloc {

struct HipPinnedAllocator {
  void* allocate(std::size_t bytes)
  {
    void* ptr = nullptr;
    hipError_t error = ::hipHostMalloc(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != hipSuccess) {
      UMPIRE_ERROR("hipMallocHost( bytes = " << bytes
                                             << " ) failed with error: "
                                             << hipGetErrorString(error));
    } else {
      return ptr;
    }
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    hipError_t error = ::hipHostFree(ptr);
    if (error != hipSuccess) {
      UMPIRE_ERROR("hipFreeHost( ptr = " << ptr << " ) failed with error: "
                                         << hipGetErrorString(error));
    }
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_HipPinnedAllocator_HPP
