//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipMallocManagedAllocator_HPP
#define UMPIRE_HipMallocManagedAllocator_HPP

#include "hip/hip_runtime_api.h"
#include "umpire/alloc/HipAllocator.hpp"
#include "umpire/config.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses hipMallocManaged and hipFree to allocate and deallocate
 *        unified memory on AMD GPUs.
 */
struct HipMallocManagedAllocator : HipAllocator {
  using HipAllocator::HipAllocator;

  /*!
   * \brief Allocate bytes of memory using hipMallocManaged.
   *
   * \param bytes Number of bytes to allocate.
   *
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::runtime_error if memory cannot be allocated.
   */
  void* allocate(std::size_t bytes)
  {
    void* ptr{nullptr};

    hipError_t error = ::hipMallocManaged(&ptr, bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ptr);
    if (error != hipSuccess) {
      if (error == hipErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error, fmt::format("hipMallocManaged( bytes = {} ) failed with error: {}", bytes,
                                                      hipGetErrorString(error)));
      } else {
        UMPIRE_ERROR(runtime_error, fmt::format("hipMallocManaged( bytes = {} ) failed with error: {}", bytes,
                                                hipGetErrorString(error)));
      }
    }

#ifdef UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
    if (m_granularity == MemoryResourceTraits::granularity_type::coarse_grained) {
      int device;

      hipError_t error = ::hipGetDevice(&device);
      if (error != hipSuccess) {
        UMPIRE_ERROR(runtime_error, fmt::format("hipGetDevice failed with error: {}", hipGetErrorString(error)));
      }

      UMPIRE_LOG(Debug, "::hipMemAdvise(hipMemAdviseSetCoarseGrain)");
      error = ::hipMemAdvise(ptr, bytes, hipMemAdviseSetCoarseGrain, device);

      if (error != hipSuccess) {
        UMPIRE_ERROR(runtime_error,
                     fmt::format("hipMemAdvise( src_ptr = {}, length = {}, device = {}) failed with error: {}", ptr,
                                 bytes, device, hipGetErrorString(error)));
      }
    }
#endif // UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY

    return ptr;
  }

  /*!
   * \brief Deallocate memory using hipFree.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::runtime_error if memory be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

    hipError_t error = ::hipFree(ptr);
    if (error != hipSuccess) {
      UMPIRE_ERROR(runtime_error,
                   fmt::format("hipFree( ptr = {} ) failed with error: {}", ptr, hipGetErrorString(error)));
    }
  }

  bool isAccessible(Platform p)
  {
    if (p == Platform::hip || p == Platform::host)
      return true;
    else
      return false;
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_hipMallocManagedAllocator_HPP
