//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MallocAllocator_HPP
#define UMPIRE_MallocAllocator_HPP

#include <cstdlib>

#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"

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
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(std::size_t bytes)
  {
    void* ret = ::malloc(bytes);
    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << ret);

    if (ret == nullptr) {
      UMPIRE_ERROR("malloc( bytes = " << bytes << " ) failed");
    } else {
      return ret;
    }
  }

  /*!
   * \brief Deallocate memory using free.
   *
   * \param ptr Address to deallocate.
   *
   * \throws umpire::util::Exception if memory cannot be free'd.
   */
  void deallocate(void* ptr)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
    ::free(ptr);
  }

  bool isHostPageable()
  {
#if defined(UMPIRE_ENABLE_HIP)
    hipDeviceProp_t props;
    int hdev = 0; //TODO: fix this 
    hipGetDevice(&hdev);

    //Check whether HIP can map host memory.
    hipGetDeviceProperties(&props, hdev);
    if(props.canMapHostMemory)
      return true;
    else
      return false;
#endif
#if defined(UMPIRE_ENABLE_CUDA)
    int pageableMem = 0;
    int cdev = 0; //TODO: fix this
    cudaGetDevice(&cdev);

    //Device supports coherently accessing pageable memory
    //without calling cudaHostRegister on it
    cudaDeviceGetAttribute(&pageableMem,
              cudaDevAttrPageableMemoryAccess, cdev);
    if(pageableMem)
      return true;
    else
      return false;
#endif
    return false; //shouldn't reach this
  }

  bool isAccessible(Platform p)
  {
    if(p == Platform::host || p == Platform::omp_target)
      return true;
    else if(p == Platform::cuda || p == Platform::hip)
      return isHostPageable();
    else  
      return false;
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_MallocAllocator_HPP
