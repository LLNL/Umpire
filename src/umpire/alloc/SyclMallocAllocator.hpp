//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SyclMallocAllocator_HPP
#define UMPIRE_SyclMallocAllocator_HPP

#include <CL/sycl.hpp>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses sycl's malloc and free to allocate and deallocate memory on
 *        Intel GPUs.
 */
struct SyclMallocAllocator {
  /*!
   * \Brief Allocate bytes of memory using SYCL malloc
   *
   * \param size Number of bytes to allocate.
   * \param queue_t SYCL queue for providing information on device and context
   * \return Pointer to start of the allocation on device.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(std::size_t size, const cl::sycl::queue& queue_t)
  {
    void* ptr = cl::sycl::malloc_device(size, queue_t);

    UMPIRE_LOG(Debug, "(bytes=" << size << ") returning " << ptr);

    if (ptr == nullptr) {
      UMPIRE_ERROR("SYCL malloc_device( bytes = " << size
                                                  << " ) failed with error!");
    } else {
      return ptr;
    }
  }

  /*!
   * \brief Deallocate memory using SYCL free.
   *
   * \param ptr Address to deallocate.
   * \param queue_t SYCL queue this pointer was asociated with
   *
   * \throws umpire::util::Exception if memory cannot be free'd.
   */
  void deallocate(void* ptr, const cl::sycl::queue& queue_t)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

    cl::sycl::free(ptr, queue_t);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_SyclMallocAllocator_HPP
