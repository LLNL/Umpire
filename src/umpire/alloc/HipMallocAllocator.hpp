//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipMallocAllocator_HPP
#define UMPIRE_HipMallocAllocator_HPP

#include <hip/hip_runtime_api.h>

#include "umpire/alloc/HipAllocator.hpp"
#include "umpire/config.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses hipMalloc and hipFree to allocate and deallocate memory on
 *        AMD GPUs.
 */
struct HipMallocAllocator : HipAllocator {
  using HipAllocator::HipAllocator;

  /*!
   * \brief Allocate bytes of memory using hipMalloc
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::runtime_error if memory cannot be allocated.
   */
  void* allocate(std::size_t size)
  {
    hipError_t error;
    void* ptr{nullptr};

    switch (m_granularity) {
      default:
      case MemoryResourceTraits::granularity_type::unknown:
        UMPIRE_LOG(Debug, "::hipMalloc(" << size << ")");
        error = ::hipMalloc(&ptr, size);
        break;

      case MemoryResourceTraits::granularity_type::fine_grained:
#ifdef UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
        UMPIRE_LOG(Debug, "::hipMallocWithFlags(" << size << ", hipDeviceMallocFinegrained)");
        error = ::hipExtMallocWithFlags(&ptr, size, hipDeviceMallocFinegrained);
#else
        UMPIRE_ERROR(runtime_error, fmt::format("Fine grained memory coherence not supported for allocation"));
#endif // UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
        break;

      case MemoryResourceTraits::granularity_type::coarse_grained:
#ifdef UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
        UMPIRE_LOG(Debug, "::hipMallocWithFlags(" << size << ", hipDeviceMallocDefault)");
        error = ::hipExtMallocWithFlags(&ptr, size, hipDeviceMallocDefault);
#else
        UMPIRE_ERROR(runtime_error, fmt::format("Coarse grained memory coherence not supported for allocation"));
#endif // UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY
        break;
    }

    if (error != hipSuccess) {
      if (error == hipErrorMemoryAllocation) {
        UMPIRE_ERROR(out_of_memory_error, fmt::format("hipExtMallocWithFlags( bytes = {} ) failed with error: {}", size,
                                                      hipGetErrorString(error)));
      } else {
        UMPIRE_ERROR(runtime_error, fmt::format("hipExtMallocWithFlags( bytes = {} ) failed with error: {}", size,
                                                hipGetErrorString(error)));
      }
    }

    UMPIRE_LOG(Debug, "(bytes=" << size << ") returning " << ptr);

    return ptr;
  }

  /*!
   * \brief Deallocate memory using hipFree.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::runtime_error if memory cannot be free'd.
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
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_HipMallocAllocator_HPP
