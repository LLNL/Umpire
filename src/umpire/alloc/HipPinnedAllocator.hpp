//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipPinnedAllocator_HPP
#define UMPIRE_HipPinnedAllocator_HPP

#include <hip/hip_runtime.h>

#include "umpire/alloc/HipAllocator.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

struct HipPinnedAllocator : HipAllocator {
  void* allocate(std::size_t bytes)
  {
    hipError_t error;
    void* ptr{nullptr};

    switch (m_granularity) {
      default:
      case umpire::strategy::GranularityController::Granularity::Default:
        error = ::hipHostMalloc(&ptr, bytes);
        break;

      case umpire::strategy::GranularityController::Granularity::FineGrainedCoherence:
        error = ::hipHostMalloc(&ptr, bytes, hipDeviceMallocFinegrained);
        break;

      case umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence:
        error = ::hipHostMalloc(&ptr, bytes, hipDeviceMallocDefault);
        break;
    }

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
