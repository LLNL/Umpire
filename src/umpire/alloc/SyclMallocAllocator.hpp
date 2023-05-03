//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SyclMallocAllocator_HPP
#define UMPIRE_SyclMallocAllocator_HPP

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/sycl_compat.hpp"

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
   * \throws umpire::util::runtime_error if memory cannot be allocated.
   */
  void* allocate(std::size_t size, const sycl::queue& queue_t)
  {
    void* ptr = sycl::malloc_device(size, queue_t);

    UMPIRE_LOG(Debug, "(bytes=" << size << ") returning " << ptr);

    if (ptr == nullptr) {
      UMPIRE_ERROR(runtime_error, umpire::fmt::format("sycl::malloc_device( bytes = {} ) failed", size));
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
   * \throws umpire::util::runtime_error if memory cannot be free'd.
   */
  void deallocate(void* ptr, const sycl::queue& queue_t)
  {
    UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

    sycl::free(ptr, queue_t);
  }

  bool isAccessible(umpire::Platform p)
  {
    if (p == umpire::Platform::sycl)
      return true;
    else
      return false;
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_SyclMallocAllocator_HPP
