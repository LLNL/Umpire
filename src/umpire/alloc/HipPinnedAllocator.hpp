//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#ifndef UMPIRE_HipPinnedAllocator_HPP
#define UMPIRE_HipPinnedAllocator_HPP

#include <hip/hip_runtime.h>

namespace umpire {
namespace alloc {

struct HipPinnedAllocator
{
  void* allocate(size_t bytes)
  {
    void* ptr = nullptr;
    hipError_t error = ::hipHostMalloc(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != hipSuccess) {
      UMPIRE_ERROR("hipMallocHost( bytes = " << bytes << " ) failed with error: " << hipGetErrorString(error));
    } else {
      return ptr;
    }
  }

  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    hipError_t error = ::hipHostFree(ptr);
    if (error != hipSuccess) {
      UMPIRE_ERROR("hipFreeHost( ptr = " << ptr << " ) failed with error: " << hipGetErrorString(error));
    }
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_HipPinnedAllocator_HPP
