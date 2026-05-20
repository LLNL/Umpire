//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MallocAllocator_HPP
#define UMPIRE_MallocAllocator_HPP

#include <cerrno>
#include <cstdlib>

#include "umpire/util/Platform.hpp"
#include "umpire/util/error.hpp"

#if defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif
#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace umpire {
namespace alloc {

/*!
 * \brief Uses malloc and free to allocate and deallocate CPU memory.
 */
struct MallocAllocator {
  /*!
   * \brief Allocate bytes of memory using malloc.
   *
   * \param bytes Number of bytes to allocate.
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::runtime_error if memory cannot be allocated.
   */
  void* allocate(std::size_t bytes)
  {
    void* ret = ::malloc(bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

    if (ret == nullptr) {
      if (errno == ENOMEM) {
        UMPIRE_ERROR(out_of_memory_error, fmt::format("malloc( bytes = {} ) failed.", bytes))
      } else {
        UMPIRE_ERROR(runtime_error, fmt::format("malloc( bytes = {} ) failed {}", bytes, strerror(errno)))
      }
    }

    return ret;
  }

  /*!
   * \brief Deallocate memory using free.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::runtime_error if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    ::free(ptr);
  }

  bool isHostPageable()
  {
#if defined(UMPIRE_ENABLE_CUDA)
    int pageableMem = 0;
    int cdev = 0;
    cudaError_t err = cudaGetDevice(&cdev);
    if (err != cudaSuccess) {
      UMPIRE_ERROR(umpire::runtime_error, fmt::format("cudaGetDevice failed with error: {}", cudaGetErrorString(err)));
    }

    // Device supports coherently accessing pageable memory
    // without calling cudaHostRegister on it
    err = cudaDeviceGetAttribute(&pageableMem, cudaDevAttrPageableMemoryAccess, cdev);
    if (err != cudaSuccess) {
      UMPIRE_ERROR(
          runtime_error,
          fmt::format("cudaDeviceGetAttribute(pageableMem = {}, cudaDevAttrPageableMemoryAccess = {}, cdev = "
                      "{}) failed with error: {}",
                      pageableMem, static_cast<int>(cudaDevAttrPageableMemoryAccess), cdev, cudaGetErrorString(err)));
    }
    if (pageableMem)
      return true;
#endif
    return false;
  }

  bool isAccessible(Platform p)
  {
    if (p == Platform::host || p == Platform::omp_target)
      return true;
    else if (p == Platform::cuda)
      return isHostPageable();
    else
      return false;
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_MallocAllocator_HPP
