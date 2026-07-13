//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SyclMallocManagedAllocator_HPP
#define UMPIRE_SyclMallocManagedAllocator_HPP

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/sycl_compat.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses sycl_shared and sycl_free to allocate and deallocate
 *        unified shared memory (USM) on Intel GPUs.
 */
struct SyclMallocManagedAllocator {
  /*!
   * \brief Allocate bytes of memory using sycl::malloc_shared.
   *
   * \param bytes Number of bytes to allocate.
   *
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::runtime_error if memory cannot be allocated.
   */
  void* allocate(std::size_t bytes, const sycl::queue& queue_t)
  {
    void* usm_ptr = sycl::malloc_shared(bytes, queue_t);

    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << usm_ptr);

    if (usm_ptr == nullptr) {
      UMPIRE_ERROR(runtime_error, fmt::format("sycl::malloc_shared( bytes = {} ) failed", bytes));
    } else {
      return usm_ptr;
    }
  }

  /*!
   * \brief Deallocate memory using sycl::free.
   *
   * \param usm_ptr Address to deallocate.
   *
   * \throws umpire::util::runtime_error if memory be free'd.
   */
  void deallocate(void* usm_ptr, const sycl::queue& queue_t)
  {
    UMPIRE_LOG(Debug, "(usm_ptr=" << usm_ptr << ")");

    sycl::free(usm_ptr, queue_t);
  }

  bool isAccessible(Platform p)
  {
    if (p == Platform::sycl || p == Platform::host)
      return true;
    else
      return false; // p is undefined
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_SyclMallocManagedAllocator_HPP
