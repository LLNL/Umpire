//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SyclMallocManagedAllocator_HPP
#define UMPIRE_SyclMallocManagedAllocator_HPP

#include <CL/sycl.hpp>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace alloc {

/*!
 * \brief Uses sycl_shared and sycl_free to allocate and deallocate
 *        unified shared memory (USM) on Intel GPUs.
 */
struct SyclMallocManagedAllocator {
  /*!
   * \brief Allocate bytes of memory using cl::sycl::malloc_shared.
   *
   * \param bytes Number of bytes to allocate.
   *
   * \return Pointer to start of the allocation.
   *
   * \throws umpire::util::Exception if memory cannot be allocated.
   */
  void* allocate(std::size_t bytes, const cl::sycl::queue& queue_t)
  {
    void* usm_ptr = cl::sycl::malloc_shared(bytes, queue_t);

    UMPIRE_LOG(Debug, "(bytes=" << bytes << ") returning " << usm_ptr);

    if (usm_ptr == nullptr) {
      UMPIRE_ERROR("cl::sycl::malloc_shared( bytes = "
                   << bytes << " ) failed with error!");
    } else {
      return usm_ptr;
    }
  }

  /*!
   * \brief Deallocate memory using cl::sycl::free.
   *
   * \param usm_ptr Address to deallocate.
   *
   * \throws umpire::util::Exception if memory be free'd.
   */
  void deallocate(void* usm_ptr, const cl::sycl::queue& queue_t)
  {
    UMPIRE_LOG(Debug, "(usm_ptr=" << usm_ptr << ")");

    cl::sycl::free(usm_ptr, queue_t);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_SyclMallocManagedAllocator_HPP
